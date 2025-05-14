use std::{
    alloc::Layout,
    cmp,
    collections::{hash_map, HashMap},
    ptr::NonNull,
    slice,
    sync::atomic::{AtomicU64, Ordering},
};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation as GpuAllocation, AllocationCreateDesc as GpuAllocationCreateDesc,
};

use crate::{
    buffer::{BufferKey, BufferSyncState},
    device::{OwnerId, SubmissionId},
    misc::timeout_u64,
    Device,
};

const INIT_COPY_OP_CAPACITY: usize = 64;

pub struct UploadDst {
    handle: vk::Buffer,
    key: BufferKey,
    offset: u64,
}

#[derive(Clone, Debug)]
pub struct CapacityError;

#[derive(Clone)]
pub(crate) struct UploadBufferInfo {
    pub label: String,
    pub size: u64,
}

pub(crate) struct UploadBuffer {
    info: UploadBufferInfo,

    // Buffer handle.
    handle: vk::Buffer,
    // Backing memory.
    _mem: GpuAllocation,

    // Position in the buffer since the previous flush, in bytes.
    cursor: usize,
    // The mapped buffer memory.
    mapped: NonNull<u8>,

    // Pending buffer copy operations.
    buffer_copies: Vec<BufferCopy>,
    // List of ending indices of each buffer batch.
    //
    // This is populated at submission time after sorting `buffer_copies`.
    buffer_batches: Vec<u32>,
    // Scratch space for building buffer barrier list.
    buffer_barrier_scratch: Vec<vk::BufferMemoryBarrier2<'static>>,
    // Scratch space for building buffer copy list.
    buffer_copy_scratch: Vec<vk::BufferCopy2<'static>>,

    // Map from buffer key to initial sync state.
    //
    // When an upload to a buffer is scheduled, the upload pool takes ownership of the destination
    // buffer, and its initial sync state is stored here. When all scheduled uploads to that buffer
    // are completed, the entry is removed, and ownership is released.
    buffer_ownership: HashMap<BufferKey, BufferSyncState>,

    // Command pool backing `cmd_buf`.
    cmd_pool: vk::CommandPool,
    // Command buffer for recording transfer operations.
    cmd_buf: vk::CommandBuffer,

    // Timeline semaphore tracking submission status.
    timeline: vk::Semaphore,
    // Timeline value of the next submission.
    timeline_value: AtomicU64,

    // Unique owner ID of this upload buffer.
    owner_id: OwnerId,
}

// Needed because UploadBuffer contains a `NonNull<u8>` pointing to the mapped memory. That memory
// is exclusively owned by this type, making these safe to implement.
unsafe impl Send for UploadBuffer {}
unsafe impl Sync for UploadBuffer {}

impl UploadBuffer {
    fn create(device: &Device, info: &UploadBufferInfo) -> UploadBuffer {
        let info = info.clone();

        // Create the buffer object.
        let buffer_info = vk::BufferCreateInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .size(info.size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // Ignored due to exclusive sharing mode.
            .queue_family_indices(&[]);
        let buffer = unsafe {
            device
                .ash_device()
                .create_buffer(&buffer_info, None)
                .unwrap()
        };

        // Allocate and bind the backing memory for the buffer.
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem = device
            .allocate(&GpuAllocationCreateDesc {
                name: &info.label,
                requirements,
                // TODO(dp): gpu_allocator doesn't accept enough detail to pick a good memory
                // type here. On RDNA, this will almost certainly allocate from the tiny pool of
                // host-local, device-visible memory, which is only 256 MB. That makes it
                // impossible to saturate the PCI bus. Should swap gpu_allocator out for VMA so we
                // have explicit control over which heap gets allocated from.
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        unsafe { device.bind_buffer_memory(buffer, mem.memory(), 0).unwrap() };

        let mapped: NonNull<u8> = mem.mapped_ptr().unwrap().cast();

        // Create a command pool and a single command buffer.
        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.upload_queue().family().as_u32());
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info).unwrap() };
        let cmd_buf_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info).unwrap()[0] };

        // Create a timeline semaphore.
        let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline_value = 1;
        let sem_info = vk::SemaphoreCreateInfo::default()
            .flags(vk::SemaphoreCreateFlags::empty())
            .push_next(&mut sem_type_info);
        let timeline = unsafe { device.create_semaphore(&sem_info).unwrap() };

        UploadBuffer {
            info,
            handle: buffer,
            _mem: mem,
            cursor: 0,
            mapped,
            cmd_pool,
            cmd_buf,
            buffer_copies: Vec::with_capacity(INIT_COPY_OP_CAPACITY),
            buffer_batches: Vec::with_capacity(INIT_COPY_OP_CAPACITY),
            buffer_barrier_scratch: Vec::with_capacity(INIT_COPY_OP_CAPACITY),
            buffer_copy_scratch: Vec::with_capacity(INIT_COPY_OP_CAPACITY),
            timeline,
            timeline_value: timeline_value.into(),
            owner_id: OwnerId::new(),
            buffer_ownership: HashMap::new(),
        }
    }

    unsafe fn range_mut(&mut self, layout: Layout, dst: UploadDst) -> Option<UploadBufferRangeMut> {
        let size = layout.size();
        let align = layout.align();

        debug_assert_eq!(align.count_ones(), 1);

        // Align the cursor up to the correct alignment. The alignment is guaranteed to be nonzero
        // and power-of-two.
        let cursor = self.cursor;
        let cursor_rem = cursor & (align - 1);
        let aligned_cursor = if cursor_rem != 0 {
            (cursor - cursor_rem).checked_add(align)?
        } else {
            cursor
        };

        // Ensure the layout fits.
        let end = aligned_cursor.checked_add(size)?;
        if u64::try_from(end).ok()? > self.info.size {
            // Doesn't fit.
            return None;
        }

        // Update the cursor.
        let src_offset = u64::try_from(cursor).ok()?;
        self.cursor = cursor;

        // Offset to the cursor.
        let ofs = isize::try_from(self.cursor).ok()?;
        let slice = unsafe {
            let ptr = self.mapped.as_ptr().offset(ofs);
            slice::from_raw_parts_mut(ptr, size)
        };

        Some(UploadBufferRangeMut {
            slice,
            src_offset,
            dst,
        })
    }

    fn prepare_owned_buffer(&mut self, buffer: BufferKey, init_state: BufferSyncState) {
        match self.buffer_ownership.entry(buffer) {
            hash_map::Entry::Occupied(o) => {
                debug_assert_eq!(o.get(), &init_state);
            }

            hash_map::Entry::Vacant(v) => {
                v.insert(init_state);
            }
        }
    }
}

/// Description of a scheduled buffer copy operation.
struct BufferCopy {
    /// Key of the destination buffer.
    dst_key: BufferKey,
    /// Handle to the destination buffer.
    dst_handle: vk::Buffer,

    /// Offset in bytes of the start of the source range.
    src_offset: u64,
    /// Offset in bytes of the start of the destination range.
    dst_offset: u64,
    /// Size in bytes of the copied range.
    size: u64,
}

impl BufferCopy {
    fn batch_order(&self, other: &Self) -> cmp::Ordering {
        // Compare key first, then destination offset.
        BufferKey::batch_order(self.dst_key, other.dst_key)
            .then(self.dst_offset.cmp(&other.dst_offset))
    }
}

/// A writable range of an upload buffer.
struct UploadBufferRangeMut<'buf> {
    slice: &'buf mut [u8],

    // Offset of `self.slice` from the start of the buffer.
    src_offset: u64,

    dst: UploadDst,
}

impl<'buf> UploadBufferRangeMut<'buf> {
    fn copy_from_slice(self, src: &[u8]) -> BufferCopy {
        self.slice.copy_from_slice(src);

        BufferCopy {
            dst_handle: self.dst.handle,
            dst_key: self.dst.key,
            src_offset: self.src_offset,
            dst_offset: self.dst.offset,
            size: src.len().try_into().unwrap(),
        }
    }
}

/// Creation parameters for an [`UploadPool`].
#[derive(Clone)]
pub struct UploadPoolInfo {
    pub label: String,
    pub num_buffers: u32,
    pub buffer_size: u64,
}

/// A pool of buffers used to upload data from the host to the GPU.
///
/// This type is used for high-throughput transfer from host memory to dedicated GPU memory. This is
/// ideal for large volumes of immutable data, like geometry and textures.
pub struct UploadPool {
    device: Device,

    buffers: Vec<UploadBuffer>,
    available: Vec<vk::Fence>,

    current: Option<usize>,
}

impl UploadPool {
    /// Creates a new upload pool.
    pub fn create(device: &Device, pool_info: &UploadPoolInfo) -> UploadPool {
        let info = pool_info.clone();

        let create_buffer = |idx| {
            let buf_info = UploadBufferInfo {
                label: format!("{}.buffers[{idx}]", info.label),
                size: info.buffer_size,
            };

            UploadBuffer::create(device, &buf_info)
        };

        let buffers = (0..info.num_buffers).map(create_buffer).collect();

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let available = unsafe {
            device
                .create_fences(&fence_info, info.num_buffers as usize)
                .unwrap()
        };

        let mut pool = UploadPool {
            device: device.clone(),
            buffers,
            available,
            current: None,
        };

        let idx = pool.wait_any_available();
        pool.begin_buffer(idx);
        pool.current = Some(idx);

        pool
    }

    /// Returns `true` if the buffer at index `idx` is available.
    fn is_available(&self, idx: usize) -> bool {
        unsafe { self.device.get_fence_status(self.available[idx]).unwrap() }
    }

    /// Waits for an upload buffer to become available.
    ///
    /// Returns the index of the first available buffer.
    fn wait_any_available(&self) -> usize {
        unsafe {
            self.device
                .wait_for_fences(&self.available, false, timeout_u64(None))
                .unwrap();
        }

        self.available
            .iter()
            .enumerate()
            .find_map(|(idx, &fence)| unsafe {
                self.device.get_fence_status(fence).unwrap().then_some(idx)
            })
            .unwrap()
    }

    fn begin_buffer(&mut self, idx: usize) {
        assert!(self.current.is_none());
        assert!(self.is_available(idx));

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .reset_command_pool(
                    self.buffers[idx].cmd_pool,
                    vk::CommandPoolResetFlags::empty(),
                )
                .unwrap();
            self.device
                .begin_command_buffer(self.buffers[idx].cmd_buf, &begin_info)
                .unwrap();
        }
    }

    /// Submits all scheduled upload operations for execution by the device.
    pub fn submit(&mut self) {
        let idx = self.current.take().unwrap();
        let buffer = &mut self.buffers[idx];

        if buffer.buffer_copies.is_empty() {
            // Nothing to do.
            return;
        }

        // Prepare scratch space.
        let num_buf_copies = buffer.buffer_copies.len();

        buffer.buffer_batches.clear();
        buffer.buffer_batches.reserve(num_buf_copies);

        buffer.buffer_barrier_scratch.clear();
        buffer.buffer_barrier_scratch.reserve(num_buf_copies);

        buffer.buffer_copy_scratch.clear();
        buffer.buffer_copy_scratch.reserve(num_buf_copies);

        // Sort the buffer copy operations.
        //
        // The comparison op produces a list batched by buffer key, with each batch in ascending
        // order by destination offset.
        buffer
            .buffer_copies
            .sort_unstable_by(BufferCopy::batch_order);

        // Build the list of buffer copy batches. While we're at it, verify that the destination
        // ranges don't overlap.
        //
        // TODO(dp): replace windows() with array_windows() when stabilized
        for (idx, win) in buffer.buffer_copies.windows(2).enumerate() {
            let b = &win[1];
            let a = &win[0];

            if a.dst_key != b.dst_key {
                buffer.buffer_batches.push(idx as u32 + 1);
            } else {
                // Unchecked arithmetic is safe here because the public APIs do the checking at
                // insertion time.
                //
                // TODO(dp): proper error
                assert!(a.dst_offset + a.size <= b.dst_offset);
            }
        }
        buffer
            .buffer_batches
            .push(buffer.buffer_copies.len() as u32);

        // Generate buffer memory barriers.
        let mut batch_start = 0;
        for &batch_end in &buffer.buffer_batches {
            let batch_end = batch_end as usize;

            // Only look up sync state once per batch.
            let init_state = buffer
                .buffer_ownership
                .get(&buffer.buffer_copies[batch_start].dst_key)
                .expect("missing ownership data");

            // TODO(dp): this may not be able to elide bounds checks
            let batch = &buffer.buffer_copies[batch_start..batch_end];
            for copy in batch {
                buffer.buffer_barrier_scratch.push(
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(init_state.stage_mask)
                        .src_access_mask(init_state.access_mask)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .buffer(copy.dst_handle)
                        .offset(copy.dst_offset)
                        .size(copy.size),
                );
            }

            batch_start = batch_end;
        }

        // Emit a pipeline barrier containing all generated memory barriers.

        let dep_info = vk::DependencyInfo::default()
            .dependency_flags(vk::DependencyFlags::empty())
            .buffer_memory_barriers(&buffer.buffer_barrier_scratch)
            .image_memory_barriers(&[]);

        unsafe { self.device.cmd_pipeline_barrier2(buffer.cmd_buf, &dep_info) };

        // Record all copy operations.
        //
        // This proceeds in batches, with all regions of a particular destination buffer being
        // recorded in the same command.

        // Generate buffer memory barriers.
        let mut batch_start = 0;
        for &batch_end in &buffer.buffer_batches {
            let batch_end = batch_end as usize;

            let batch = &buffer.buffer_copies[batch_start..batch_end];
            let dst_handle = buffer.buffer_copies[batch_start].dst_handle;

            for buf_copy in batch {
                // TODO(dp): could reuse these state structs between submits to save a tiny bit of
                // time initializing the `s_type` field.
                buffer.buffer_copy_scratch.push(
                    vk::BufferCopy2::default()
                        .src_offset(buf_copy.src_offset)
                        .dst_offset(buf_copy.dst_offset)
                        .size(buf_copy.size),
                );
            }

            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(buffer.handle)
                .dst_buffer(dst_handle)
                .regions(&buffer.buffer_copy_scratch[batch_start..batch_end]);

            unsafe { self.device.cmd_copy_buffer2(buffer.cmd_buf, &copy_info) };

            batch_start = batch_end;
        }

        // Finish recording the command buffer.
        unsafe { self.device.end_command_buffer(buffer.cmd_buf).unwrap() };
        let cmdbuf_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(buffer.cmd_buf)
            .device_mask(0)];

        // Fetch and increment the timeline value.
        let timeline_value = buffer.timeline_value.fetch_add(1, Ordering::Relaxed);

        // Signal the semaphore when the command buffer finishes executing.
        let signal_sem_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(buffer.timeline)
            .value(timeline_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit_info = vk::SubmitInfo2::default()
            .flags(vk::SubmitFlags::empty())
            .wait_semaphore_infos(&[])
            .command_buffer_infos(cmdbuf_info)
            .signal_semaphore_infos(signal_sem_info);

        let queue = self.device.upload_queue();

        unsafe {
            // Reset the availability fence for this buffer.
            self.device.reset_fences(&[self.available[idx]]).unwrap();

            // Submit the command buffer. The availability fence will be signaled when all copies
            // have completed.
            queue
                .submit2(&[submit_info], Some(self.available[idx]))
                .unwrap();
        }

        // Release ownership of the destination buffers after updating their sync state.

        // TODO(dp): add a proper interface for this, replacing the raw vkQueueSubmit2 call
        let submission = SubmissionId {
            queue,
            timeline_value,
        };

        let mut buffers = self.device.buffers_mut();

        // TODO(dp): if all these buffers are getting identical sync info, is there a way to cut
        // down on storage space?
        for (key, _old_state) in buffer.buffer_ownership.drain() {
            buffers.release(
                key,
                buffer.owner_id,
                BufferSyncState {
                    submission: Some(submission.clone()),
                    stage_mask: vk::PipelineStageFlags2::COPY,
                    access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                },
            );
        }
    }

    /// Records a buffer copy operation, returning an upload key.
    ///
    /// If there is not enough space in the active upload buffer to copy the data in `src`, this
    /// method returns a [`CapacityError`].
    pub unsafe fn copy_bytes_to_buffer(
        &mut self,
        src: &[u8],
        dst: BufferKey,
    ) -> Result<(), CapacityError> {
        let idx = match self.current {
            Some(i) => i,
            None => {
                let i = self.wait_any_available();
                self.begin_buffer(i);
                i
            }
        };

        let uploader = &mut self.buffers[idx];

        let mut buffers = self.device.buffers_mut();

        buffers.acquire(dst, uploader.owner_id);

        let init_state = buffers
            .acquire(dst, uploader.owner_id)
            .expect("buffer owned elsewhere")
            .clone();

        // Safety: upload pool owns the buffer.
        let handle = unsafe { buffers.handle(dst).expect("stale key") };

        uploader.prepare_owned_buffer(dst, init_state);

        // Align all copy ranges to 4 byte boundaries.
        let layout = Layout::from_size_align(src.len(), 4).unwrap();
        let dst = UploadDst {
            handle,
            key: dst,
            offset: 0,
        };

        // TODO(dp): this might also be an alignment error
        let range_mut = unsafe { uploader.range_mut(layout, dst).ok_or(CapacityError)? };
        let buf_copy = range_mut.copy_from_slice(src);
        uploader.buffer_copies.push(buf_copy);

        Ok(())
    }
}

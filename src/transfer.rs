use std::{
    alloc::Layout,
    ptr::NonNull,
    slice,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation as GpuAllocation, AllocationCreateDesc as GpuAllocationCreateDesc,
};

use crate::{misc::timeout_u64, Device};

pub struct UploadKey {
    // Index of the buffer within the pool.
    idx: usize,
    // Timeline semaphore value.
    value: u64,
}

pub struct UploadDst {
    handle: vk::Buffer,
    queue_family: u32,
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
    mem: GpuAllocation,

    // Position in the buffer since the previous flush, in bytes.
    cursor: usize,
    // The mapped buffer memory.
    mapped: NonNull<u8>,

    // Pending buffer copy operations.
    buffer_copies: Vec<BufferCopy>,

    // Command pool backing `cmd_buf`.
    cmd_pool: vk::CommandPool,
    // Command buffer for recording transfer operations.
    cmd_buf: vk::CommandBuffer,

    // Timeline semaphore tracking submission status.
    timeline: vk::Semaphore,
    // Timeline value of the next submission.
    timeline_value: AtomicU64,
}

impl UploadBuffer {
    fn create(device: &Device, info: &UploadBufferInfo) -> UploadBuffer {
        let info = info.clone();

        let buffer_info = vk::BufferCreateInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .size(info.size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // Ignored due to exclusive sharing mode.
            .queue_family_indices(&[]);

        let buffer = unsafe { device.create_buffer(&buffer_info).unwrap() };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mem = device
            .allocate(&GpuAllocationCreateDesc {
                // TODO: uniquely name upload buffers
                name: "upload_buffer",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe { device.bind_buffer_memory(buffer, mem.memory(), 0).unwrap() };

        let mapped: NonNull<u8> = mem.mapped_ptr().unwrap().cast();

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.queue_family_index());
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info).unwrap() };
        let cmd_buf_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_info).unwrap()[0] };

        let timeline_value = 0;
        let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(timeline_value);
        let sem_info = vk::SemaphoreCreateInfo::default()
            .flags(vk::SemaphoreCreateFlags::empty())
            .push_next(&mut sem_type_info);
        let timeline = unsafe { device.create_semaphore(&sem_info).unwrap() };

        UploadBuffer {
            info,
            handle: buffer,
            mem,
            cursor: 0,
            mapped,
            cmd_pool,
            cmd_buf,
            buffer_copies: Vec::new(),
            timeline,
            timeline_value: timeline_value.into(),
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
}

/// Description of a scheduled buffer copy operation.
struct BufferCopy {
    /// Handle to the destination buffer.
    dst_handle: vk::Buffer,
    /// Index of the destination queue family.
    dst_queue_family: u32,

    /// Offset of the start of the source range.
    src_offset: u64,
    /// Offset of the start of the destination range.
    dst_offset: u64,
    /// Size of the copied range.
    size: u64,
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
            dst_queue_family: self.dst.queue_family,
            src_offset: self.src_offset,
            dst_offset: self.dst.offset,
            size: src.len().try_into().unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct UploadPoolInfo {
    pub label: String,
    pub num_buffers: u32,
    pub buffer_size: u64,
}

pub struct UploadPool {
    buffers: Vec<UploadBuffer>,
    available: Vec<vk::Fence>,

    current: Option<usize>,
}

impl UploadPool {
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
            buffers,
            available,
            current: None,
        };

        let idx = pool.wait_any_available(device);
        pool.begin_buffer(device, idx);

        pool
    }

    /// Returns `true` if the buffer at index `idx` is available.
    fn is_available(&self, device: &Device, idx: usize) -> bool {
        unsafe { device.get_fence_status(self.available[idx]).unwrap() }
    }

    /// Waits for an upload buffer to become available.
    ///
    /// Returns the index of the first available buffer.
    fn wait_any_available(&self, device: &Device) -> usize {
        unsafe {
            device
                .wait_for_fences(&self.available, false, timeout_u64(None))
                .unwrap();
        }

        self.available
            .iter()
            .enumerate()
            .find_map(|(idx, &fence)| unsafe {
                device.get_fence_status(fence).unwrap().then_some(idx)
            })
            .unwrap()
    }

    fn begin_buffer(&mut self, device: &Device, idx: usize) {
        assert!(self.current.is_none());
        assert!(self.is_available(device, idx));

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(self.buffers[idx].cmd_buf, &begin_info)
                .unwrap();
        }
    }

    fn submit(&mut self, device: &Device) {
        let idx = self.current.take().unwrap();
        let buffer = &mut self.buffers[idx];

        // TODO(dp): group by buffer handle to emit multi-region barriers
        let mut buffer_barriers = Vec::new();
        for buf_copy in buffer.buffer_copies.drain(..) {
            let copy_regions = &[vk::BufferCopy2::default()
                .src_offset(buf_copy.src_offset)
                .dst_offset(buf_copy.dst_offset)
                .size(buf_copy.size)];
            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(buffer.handle)
                .dst_buffer(buf_copy.dst_handle)
                .regions(copy_regions);

            unsafe { device.cmd_copy_buffer2(buffer.cmd_buf, &copy_info) };

            if buf_copy.dst_queue_family == device.transfer_queue_family_index() {
                // Barrier would have no effect.
                continue;
            }

            // Release the affected buffers to their respective destination queues.
            //
            // The barriers don't need stage or access masks; the semaphore signal operation orders
            // all prior commands and memory accesses before it, and subsequent commands operating
            // on the affected resources are required to perform a semaphore wait operation which
            // orders those commands and accesses after it.
            buffer_barriers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::empty())
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_access_mask(vk::AccessFlags2::empty())
                    // TODO
                    .src_queue_family_index(device.transfer_queue_family_index())
                    .dst_queue_family_index(buf_copy.dst_queue_family)
                    .buffer(buf_copy.dst_handle)
                    .offset(buf_copy.dst_offset)
                    .size(buf_copy.size),
            );
        }

        let image_barriers = Vec::new();

        // Emit resource barriers, if any.
        if !(buffer_barriers.is_empty() && image_barriers.is_empty()) {
            let dep_info = vk::DependencyInfo::default()
                .dependency_flags(vk::DependencyFlags::empty())
                .memory_barriers(&[])
                .buffer_memory_barriers(&buffer_barriers)
                // TODO
                .image_memory_barriers(&image_barriers);

            unsafe { device.cmd_pipeline_barrier2(buffer.cmd_buf, &dep_info) };
        }

        // Finish recording the command buffer.
        unsafe { device.end_command_buffer(buffer.cmd_buf).unwrap() };
        let cmdbuf_info = &[vk::CommandBufferSubmitInfo::default()
            .command_buffer(buffer.cmd_buf)
            .device_mask(0)];

        // Increment the timeline value.
        let timeline_value = buffer.timeline_value.fetch_add(1, Ordering::Relaxed);
        let signal_sem_info = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(buffer.timeline)
            .value(timeline_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];

        let submit_info = vk::SubmitInfo2::default()
            .flags(vk::SubmitFlags::empty())
            .wait_semaphore_infos(&[])
            .command_buffer_infos(cmdbuf_info)
            .signal_semaphore_infos(signal_sem_info);

        unsafe {
            // Reset the availability fence for this buffer.
            device.reset_fences(&[self.available[idx]]).unwrap();

            // Submit the command buffer. The availability fence will be signaled when all transfer
            // destination resources have been released to the destination queues.
            device
                .transfer_queue()
                .submit2(&[submit_info], Some(self.available[idx]))
                .unwrap();
        }
    }

    pub unsafe fn copy_bytes_to_buffer(
        &mut self,
        device: &Device,
        src: &[u8],
        alignment: usize,
        dst: vk::Buffer,
        dst_queue_family_index: u32,
    ) -> Result<UploadKey, CapacityError> {
        let idx = match self.current {
            Some(i) => i,
            None => {
                let i = self.wait_any_available(device);
                self.begin_buffer(device, i);
                i
            }
        };

        let buf = &mut self.buffers[idx];
        let layout = Layout::from_size_align(src.len(), alignment).unwrap();
        let dst = UploadDst {
            handle: dst,
            queue_family: dst_queue_family_index,
            offset: 0,
        };

        // TODO(dp): this might also be an alignment error
        let range_mut = unsafe { buf.range_mut(layout, dst).ok_or(CapacityError)? };
        let buf_copy = range_mut.copy_from_slice(src);
        buf.buffer_copies.push(buf_copy);

        Ok(UploadKey {
            idx,
            value: buf.timeline_value.load(Ordering::Relaxed),
        })
    }
}

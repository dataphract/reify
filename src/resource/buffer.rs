use std::iter;

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation as GpuAllocation, AllocationScheme},
    MemoryLocation,
};

use crate::{
    device::{OwnerId, Ownership, SubmissionId},
    pool::{FreeList, RawPool},
    Device,
};

pub struct BufferStorageHot {
    handle: vk::Buffer,
}

pub struct BufferStorageCold {
    info: BufferInfo,
    mem: GpuAllocation,
    ownership: Ownership<BufferSyncState>,
}

/// Buffer resource metadata.
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferKey {
    index: u32,
    generation: u32,
}

impl BufferKey {
    // Intentionally not an Ord implementation -- only expose this comparison internally.
    pub(crate) fn batch_order(self, other: Self) -> std::cmp::Ordering {
        self.index
            .cmp(&other.index)
            .then(self.generation.cmp(&other.generation))
    }
}

pub struct BufferPool {
    free_list: FreeList,

    // Each buffer is protected by a generation counter. If the provided key has the wrong
    // generation, the associated buffer no longer exists.
    generation: Vec<u32>,

    // Hot storage holds only the data required by the inner draw loop.
    hot: RawPool<BufferStorageHot>,

    // Cold storage holds data used when modifying, deleting, binding, etc.
    cold: RawPool<BufferStorageCold>,
}

impl BufferPool {
    pub fn with_capacity(cap: usize) -> BufferPool {
        let free_list = FreeList::with_capacity(cap);
        let generations = iter::repeat(0).take(cap).collect();
        let hot = RawPool::with_capacity(cap);
        let cold = RawPool::with_capacity(cap);

        BufferPool {
            free_list,
            generation: generations,
            hot,
            cold,
        }
    }

    #[inline]
    fn check_key(&self, key: BufferKey) -> bool {
        key.generation == self.generation[key.index as usize]
    }

    // Attempts to set `owner` as the owner of the buffer associated with `key`.
    pub(crate) fn acquire(&mut self, key: BufferKey, owner: OwnerId) -> Option<&BufferSyncState> {
        if !self.check_key(key) {
            return None;
        }

        let cold = unsafe { self.cold.get_mut(key.index as usize).assume_init_mut() };

        Some(cold.ownership.acquire(owner))
    }

    // Safety: may only be called by the buffer's current owner.
    //
    // TODO(dp): this shouldn't be exposed in the public API. Instead, it should be exposed as a
    // safe method on NodeContext, which calls this method.
    pub unsafe fn handle(&self, key: BufferKey) -> Option<vk::Buffer> {
        let index = key.index as usize;

        if key.generation != self.generation[index] {
            return None;
        }

        // Safety: if a key with the right generation exists, the slot is initialized.
        unsafe { Some(self.hot.get(index).assume_init_ref().handle) }
    }

    // Attempts to release `owner`'s ownership of the buffer associated with `key`.
    pub(crate) fn release(&mut self, key: BufferKey, owner: OwnerId, state: BufferSyncState) {
        if !self.check_key(key) {
            // TODO(dp): error
            panic!("stale key");
        }

        let cold = unsafe { self.cold.get_mut(key.index as usize).assume_init_mut() };

        cold.ownership.release(owner, Some(state));
    }

    pub fn create(&mut self, device: &Device, info: BufferInfo) -> Option<BufferKey> {
        let index = self.free_list.pop()?;

        let buffer_create_info = vk::BufferCreateInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe {
            device
                .ash_device()
                .create_buffer(&buffer_create_info, None)
                .expect("buffer creation failed")
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };

        let allocation_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "unnamed",
            requirements,
            location: info.location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let mem = device
            .allocate(&allocation_desc)
            .expect("allocation failed");

        unsafe {
            device
                .bind_buffer_memory(handle, mem.memory(), 0)
                .expect("failed to bind buffer memory")
        };

        self.hot.get_mut(index).write(BufferStorageHot { handle });
        self.cold.get_mut(index).write(BufferStorageCold {
            info,
            mem,
            ownership: Ownership::new(BufferSyncState {
                submission: None,
                stage_mask: vk::PipelineStageFlags2::empty(),
                access_mask: vk::AccessFlags2::empty(),
            }),
        });

        Some(BufferKey {
            generation: self.generation[index],
            index: index as u32,
        })
    }

    pub fn destroy(&mut self, device: &Device, key: BufferKey) {
        let index = key.index as usize;

        if key.generation < self.generation[index] {
            panic!("already destroyed: {key:?}");
        }

        let BufferStorageHot { handle } = unsafe { self.hot.get_mut(index).assume_init_read() };
        let BufferStorageCold {
            info: _,
            mem,
            ownership,
        } = unsafe { self.cold.get_mut(index).assume_init_read() };

        if let Some(owner) = ownership.owner() {
            // TODO: get the owner label from the device, and return a proper error
            panic!("attempted to destroy resource still owned by {owner:?}");
        }

        unsafe { device.destroy_buffer(handle) };
        unsafe { device.free(mem) };

        self.generation[index] += 1;
        self.free_list.push(index);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSyncState {
    pub submission: Option<SubmissionId>,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
}

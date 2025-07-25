use ash::vk;
use vk_mem::{self as vma, Alloc};

use crate::{
    device::SubmissionId,
    resource::{Resource, ResourceKey},
    Device,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Buffer {
    key: ResourceKey,
}

impl Buffer {
    #[inline]
    pub(crate) fn batch_order(self, other: Self) -> std::cmp::Ordering {
        self.key.batch_order(other.key)
    }
}

impl Resource for Buffer {
    type CreateInfo = BufferInfo;
    type OwnerState = BufferSyncState;
    type Handle = vk::Buffer;
    type Cold = BufferStorageCold;

    #[inline]
    fn from_key(key: ResourceKey) -> Buffer {
        Buffer { key }
    }

    #[inline]
    fn key(&self) -> ResourceKey {
        self.key
    }

    fn create(device: &Device, info: Self::CreateInfo) -> (Self::Handle, Self::Cold) {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .flags(vk::BufferCreateFlags::empty())
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vma::AllocationCreateInfo {
            flags: vma::AllocationCreateFlags::empty(),
            usage: vma::MemoryUsage::Auto,
            required_flags: vk::MemoryPropertyFlags::empty(),
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: u32::MAX,
            user_data: 0,
            priority: 0.5,
        };

        let (handle, mem) = unsafe {
            device
                .allocator()
                .create_buffer(&buffer_create_info, &alloc_info)
                .expect("buffer creation and allocation failed")
        };

        let cold = BufferStorageCold { info, mem };

        (handle, cold)
    }

    fn destroy(device: &Device, handle: Self::Handle, cold: Self::Cold) {
        let BufferStorageCold { info: _, mut mem } = cold;

        unsafe { device.allocator().destroy_buffer(handle, &mut mem) };
    }
}

pub struct BufferStorageHot {
    handle: vk::Buffer,
}

pub struct BufferStorageCold {
    info: BufferInfo,
    mem: vma::Allocation,
}

/// Buffer resource metadata.
pub struct BufferInfo {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub class: BufferMemoryType,
}

pub enum BufferMemoryType {
    /// Buffer is used for download of data from GPU to host.
    Download,
    /// Buffer is used for low-volume direct upload of constant data from host to GPU.
    UploadConstant,
    /// Buffer is used for staging host data for high-volume upload to GPU.
    UploadStaging,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSyncState {
    pub submission: Option<SubmissionId>,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
}

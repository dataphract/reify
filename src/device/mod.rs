use std::{
    ffi::CStr,
    io::Write,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use ash::{
    ext, khr,
    prelude::*,
    vk::{self, Handle},
};
use gpu_allocator::vulkan::{
    Allocation as GpuAllocation, AllocationCreateDesc as GpuAllocationCreateDesc,
};

use crate::resource::buffer::{BufferInfo, BufferKey, BufferPool};

pub(crate) mod extensions;

mod features;

mod physical;
pub use physical::PhysicalDevice;

// TODO(dp): this should become an array to support multiple devices in use at once. Device loss can
// be handled with generational indices, if desired.
static DEVICE: OnceLock<DeviceStorage> = OnceLock::new();

#[derive(Clone)]
pub struct Device {}

struct DeviceStorage {
    phys_device: PhysicalDevice,

    graphics_queue: Queue,
    upload_queue: Queue,
    _download_queue: Queue,

    queues: Vec<QueueStorage>,

    ash: ash::Device,
    khr_swapchain: khr::swapchain::Device,
    ext_debug_utils: ext::debug_utils::Device,

    allocator: Mutex<gpu_allocator::vulkan::Allocator>,

    // TODO(dp): I don't like this locking scheme.
    //
    // Ideally, there should be a batch interface for ownership transfer, and retrieving data about
    // a resource you already own shouldn't involve any lock contention.
    buffers: RwLock<BufferPool>,
}

impl Device {
    pub fn ash_device(&self) -> &ash::Device {
        &self.storage().ash
    }

    pub(crate) fn buffers(&self) -> RwLockReadGuard<'_, BufferPool> {
        self.storage().buffers.read().unwrap()
    }

    pub(crate) fn buffers_mut(&mut self) -> RwLockWriteGuard<'_, BufferPool> {
        self.storage().buffers.write().unwrap()
    }

    pub fn allocate(&self, desc: &GpuAllocationCreateDesc) -> gpu_allocator::Result<GpuAllocation> {
        self.storage().allocator.lock().unwrap().allocate(desc)
    }

    #[allow(clippy::missing_safety_doc)]
    #[tracing::instrument(skip_all)]
    pub unsafe fn acquire_next_image_2(
        &self,
        acquire_info: &vk::AcquireNextImageInfoKHR,
    ) -> VkResult<(u32, bool)> {
        self.storage()
            .khr_swapchain
            .acquire_next_image2(acquire_info)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn cmd_begin_debug_utils_label(
        &self,
        command_buffer: vk::CommandBuffer,
        label_info: &vk::DebugUtilsLabelEXT,
    ) {
        self.storage()
            .ext_debug_utils
            .cmd_begin_debug_utils_label(command_buffer, label_info)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn cmd_end_debug_utils_label(&self, command_buffer: vk::CommandBuffer) {
        self.storage()
            .ext_debug_utils
            .cmd_end_debug_utils_label(command_buffer)
    }

    pub fn create_buffer(&self, info: BufferInfo) -> Option<BufferKey> {
        self.storage().buffers.write().unwrap().create(self, info)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn create_fences(
        &self,
        info: &vk::FenceCreateInfo,
        count: usize,
    ) -> VkResult<Vec<vk::Fence>> {
        let mut v = Vec::with_capacity(count);

        for _ in 0..count {
            let fence = match unsafe { self.create_fence(info) } {
                Ok(f) => f,
                Err(e) => {
                    for f in v.drain(..) {
                        unsafe { self.destroy_fence(f) };
                    }

                    return Err(e);
                }
            };

            v.push(fence);
        }

        Ok(v)
    }

    #[allow(clippy::missing_safety_doc)]
    #[tracing::instrument(name = "Device::create_swapchain", skip_all)]
    pub unsafe fn create_swapchain(
        &self,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<vk::SwapchainKHR> {
        self.storage().khr_swapchain.create_swapchain(info, None)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn get_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> VkResult<Vec<vk::Image>> {
        unsafe { self.storage().khr_swapchain.get_swapchain_images(swapchain) }
    }

    pub unsafe fn free(&self, allocation: GpuAllocation) {
        self.storage()
            .allocator
            .lock()
            .unwrap()
            .free(allocation)
            .expect("failed to free allocation");
    }

    #[tracing::instrument(name = "Device::queue_wait_idle", skip_all)]
    unsafe fn queue_wait_idle(&self, queue: vk::Queue) -> VkResult<()> {
        unsafe { DEVICE.get().unwrap().ash.queue_wait_idle(queue) }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn set_debug_utils_object_name<T: Handle>(
        &self,
        handle: T,
        name: &CStr,
    ) -> VkResult<()> {
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(name);

        unsafe {
            self.storage()
                .ext_debug_utils
                .set_debug_utils_object_name(&name_info)?;
        }

        Ok(())
    }

    /// Sets the debug label of a Vulkan object using a format string.
    ///
    /// This performs the string formatting internally using a fixed-size buffer on the stack.
    pub unsafe fn set_debug_utils_object_name_with<'a, T: Handle>(
        &self,
        handle: T,
        args: std::fmt::Arguments<'_>,
    ) -> VkResult<()> {
        let mut buf = std::io::Cursor::new([0u8; 64]);
        buf.write_fmt(args).map_err(|_| vk::Result::INCOMPLETE)?;

        // Truncate if the string fills the buffer.
        let mut arr = buf.into_inner();
        arr[arr.len() - 1] = 0;

        let name = CStr::from_bytes_until_nul(&arr[..]).unwrap();

        unsafe { self.set_debug_utils_object_name(handle, name)? };

        Ok(())
    }

    #[inline]
    fn storage(&self) -> &'static DeviceStorage {
        DEVICE.get().unwrap()
    }

    pub fn physical_device(&self) -> &'static PhysicalDevice {
        &self.storage().phys_device
    }

    #[inline]
    pub fn graphics_queue(&self) -> Queue {
        self.storage().graphics_queue.clone()
    }

    #[inline]
    pub fn upload_queue(&self) -> Queue {
        self.storage().upload_queue.clone()
    }
}

device_delegate! {
    impl Device {
        pub unsafe fn allocate_command_buffers(
            info: &vk::CommandBufferAllocateInfo,
        ) -> VkResult<Vec<vk::CommandBuffer>>;
        pub unsafe fn begin_command_buffer(
            cmdbuf: vk::CommandBuffer,
            begin_info: &vk::CommandBufferBeginInfo,
        ) -> VkResult<()>;
        pub unsafe fn bind_buffer_memory(
            buffer: vk::Buffer,
            memory: vk::DeviceMemory,
            memory_offset: vk::DeviceSize,
        ) -> VkResult<()>;
        pub unsafe fn bind_image_memory(
            image: vk::Image,
            memory: vk::DeviceMemory,
            memory_offset: vk::DeviceSize,
        ) -> VkResult<()>;
        pub unsafe fn cmd_begin_rendering(
            cmdbuf: vk::CommandBuffer,
            info: &vk::RenderingInfo,
        );
        pub unsafe fn cmd_begin_render_pass(
            cmdbuf: vk::CommandBuffer,
            pass_info: &vk::RenderPassBeginInfo,
            contents: vk::SubpassContents,
        );
        pub unsafe fn cmd_bind_pipeline(
            cmdbuf: vk::CommandBuffer,
            bind_point: vk::PipelineBindPoint,
            pipeline: vk::Pipeline,
        );
        pub unsafe fn cmd_bind_vertex_buffers(
            cmdbuf: vk::CommandBuffer,
            first_binding: u32,
            buffers: &[vk::Buffer],
            offsets: &[u64],
        );
        pub unsafe fn cmd_blit_image2(
            cmdbuf: vk::CommandBuffer,
            info: &vk::BlitImageInfo2,
        );
        pub unsafe fn cmd_copy_buffer2(cmdbuf: vk::CommandBuffer, info: &vk::CopyBufferInfo2);
        pub unsafe fn cmd_draw(
            cmdbuf: vk::CommandBuffer,
            vertex_count: u32,
            instance_count: u32,
            first_vertex: u32,
            first_instance: u32,
        );
        pub unsafe fn cmd_end_rendering(cmdbuf: vk::CommandBuffer);
        pub unsafe fn cmd_end_render_pass(cmdbuf: vk::CommandBuffer);
        pub unsafe fn cmd_pipeline_barrier(
            cmdbuf: vk::CommandBuffer,
            src_stage_mask: vk::PipelineStageFlags,
            dst_stage_mask: vk::PipelineStageFlags,
            dependendency_flags: vk::DependencyFlags,
            memory_barriers: &[vk::MemoryBarrier],
            buffer_memory_barriers: &[vk::BufferMemoryBarrier],
            image_memory_barriers: &[vk::ImageMemoryBarrier],
        );
        pub unsafe fn cmd_pipeline_barrier2(cmdbuf: vk::CommandBuffer, info: &vk::DependencyInfo);
        pub unsafe fn cmd_set_scissor(
            cmdbuf: vk::CommandBuffer,
            first_scissor: u32,
            scissors: &[vk::Rect2D],
        );
        pub unsafe fn cmd_set_viewport(
            cmdbuf: vk::CommandBuffer,
            first_viewport: u32,
            viewports: &[vk::Viewport],
        );
        pub unsafe fn end_command_buffer(cmdbuf: vk::CommandBuffer) -> VkResult<()>;
        pub unsafe fn get_buffer_memory_requirements(buffer: vk::Buffer) -> vk::MemoryRequirements;
        pub unsafe fn get_fence_status(fence: vk::Fence) -> VkResult<bool>;
        pub unsafe fn get_image_memory_requirements(image: vk::Image) -> vk::MemoryRequirements;
        /// Resets the command pool `pool`.
        ///
        /// # Safety
        ///
        /// - All command buffers allocated from `pool` must not be in the pending state.
        pub unsafe fn reset_command_pool(
            pool: vk::CommandPool,
            flags: vk::CommandPoolResetFlags,
        ) -> VkResult<()>;
        /// Sets the state of each element of `fences` to be unsignaled.
        ///
        /// # Safety
        ///
        /// - Each element of `fences` must not be currently associated with any queue command that
        ///   has not yet completed execution on that queue.
        /// - `fences` must not be empty.
        pub unsafe fn reset_fences(fences: &[vk::Fence]) -> VkResult<()>;
        /// Waits for one or more fences to enter the signaled state.
        ///
        /// # Safety
        /// - `fences` must not be empty.
        #[tracing::instrument(skip_all)]
        pub unsafe fn wait_for_fences(fences: &[vk::Fence], wait_all: bool, timeout: u64) -> VkResult<()>;
        #[tracing::instrument(skip_all)]
        pub unsafe fn wait_semaphores(info: &vk::SemaphoreWaitInfo, timeout: u64) -> VkResult<()>;
    }
}

// NOTE: Only add methods here if there is no external synchronization requirement on the device.
device_delegate_no_alloc_callbacks! {
    impl Device {
        pub unsafe fn create_command_pool(
            info: &vk::CommandPoolCreateInfo,
        ) -> VkResult<vk::CommandPool>;
        pub unsafe fn create_event(info: &vk::EventCreateInfo) -> VkResult<vk::Event>;
        pub unsafe fn create_fence(info: &vk::FenceCreateInfo) -> VkResult<vk::Fence>;
        pub unsafe fn create_framebuffer(
            info: &vk::FramebufferCreateInfo,
        ) -> VkResult<vk::Framebuffer>;
        pub unsafe fn create_graphics_pipelines(
            cache: vk::PipelineCache,
            info: &[vk::GraphicsPipelineCreateInfo],
        ) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>;
        pub unsafe fn create_image_view(info: &vk::ImageViewCreateInfo) -> VkResult<vk::ImageView>;
        pub unsafe fn create_image(info: &vk::ImageCreateInfo) -> VkResult<vk::Image>;
        pub unsafe fn create_pipeline_layout(
            info: &vk::PipelineLayoutCreateInfo,
        ) -> VkResult<vk::PipelineLayout>;
        pub unsafe fn create_render_pass(info: &vk::RenderPassCreateInfo) -> VkResult<vk::RenderPass>;
        pub unsafe fn create_semaphore(info: &vk::SemaphoreCreateInfo) -> VkResult<vk::Semaphore>;
        pub unsafe fn create_shader_module(
            info: &vk::ShaderModuleCreateInfo,
        ) -> VkResult<vk::ShaderModule>;
        pub unsafe fn destroy_buffer(buffer: vk::Buffer);
        pub unsafe fn destroy_command_pool(pool: vk::CommandPool);
        pub unsafe fn destroy_fence(fence: vk::Fence);
        pub unsafe fn destroy_framebuffer(framebuffer: vk::Framebuffer);
        pub unsafe fn destroy_image(image: vk::Image);
        pub unsafe fn destroy_image_view(view: vk::ImageView);
        pub unsafe fn destroy_pipeline(pipeline: vk::Pipeline);
        pub unsafe fn destroy_pipeline_layout(layout: vk::PipelineLayout);
        pub unsafe fn destroy_render_pass(pass: vk::RenderPass);
        pub unsafe fn destroy_semaphore(sem: vk::Semaphore);
        pub unsafe fn destroy_shader_module(module: vk::ShaderModule);
    }
}

/// Generates methods on `Device` and `DeviceInner` that delegate to `ash::Device`.
macro_rules! device_delegate {
    (
        impl Device {
            $(
            $(#[$m:meta])*
            $v:vis unsafe fn $name:ident(
                $($param:ident : $param_ty:ty),* $(,)?
            ) $(-> $ret_ty:ty)?;
            )*
        }
    ) => {
        impl Device {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { DEVICE.get().unwrap().ash.$name($($param),*) }
                }
            )*
        }

        impl DeviceStorage {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { self.ash.$name($($param),*) }
                }
            )*
        }
    };
}
use device_delegate;

/// Generates methods on `Device` that delegate to `ash::Device`, providing
/// `None` for the allocation callback parameter.
macro_rules! device_delegate_no_alloc_callbacks {
    (
        impl Device {
            $(
            $(#[$m:meta])*
            $v:vis unsafe fn $name:ident(
                $($param:ident : $param_ty:ty),* $(,)?
            ) $(-> $ret_ty:ty)?;
            )*
        }
    ) => {
        impl Device {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { DEVICE.get().unwrap().$name($($param),*) }
                }
            )*
        }

        impl DeviceStorage {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { self.ash.$name($($param,)* None) }
                }
            )*
        }
    };
}
use device_delegate_no_alloc_callbacks;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QueueFamily {
    index: u32,
}

impl QueueFamily {
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.index
    }
}

struct QueueStorage {
    semaphore: vk::Semaphore,
    timeline_value: AtomicU64,

    // NOTE: vk::Queue is Copy, so this Mutex shouldn't be exposed outside the impl block for
    // QueueStorage.
    handle: Mutex<vk::Queue>,
}

impl QueueStorage {
    unsafe fn create(device: &ash::Device, queue: vk::Queue) -> QueueStorage {
        let timeline_value = 0;
        let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(timeline_value);
        let sem_info = vk::SemaphoreCreateInfo::default()
            .flags(vk::SemaphoreCreateFlags::empty())
            .push_next(&mut sem_type_info);
        let semaphore = unsafe { device.create_semaphore(&sem_info, None).unwrap() };

        QueueStorage {
            semaphore,
            timeline_value: timeline_value.into(),
            handle: Mutex::new(queue),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Queue {
    /// The queue family to which the queue belongs.
    family: QueueFamily,
    /// The index of the queue in the device.
    index: u32,
}

impl Queue {
    #[inline]
    fn device(&self) -> Device {
        Device {}
    }

    #[inline]
    pub fn family(&self) -> QueueFamily {
        self.family
    }

    #[inline]
    fn storage(&self) -> &'static QueueStorage {
        &self.device().storage().queues[self.index as usize]
    }

    pub unsafe fn begin_debug_utils_label(&self, label_info: &vk::DebugUtilsLabelEXT) {
        let queue_guard = self.storage().handle.lock().unwrap();

        unsafe {
            self.device()
                .storage()
                .ext_debug_utils
                .queue_begin_debug_utils_label(*queue_guard, label_info)
        };

        drop(queue_guard);
    }

    pub unsafe fn end_debug_utils_label(&self) {
        let queue_guard = self.storage().handle.lock().unwrap();

        unsafe {
            self.device()
                .storage()
                .ext_debug_utils
                .queue_end_debug_utils_label(*queue_guard)
        };

        drop(queue_guard);
    }

    #[tracing::instrument(skip_all)]
    pub unsafe fn present(&self, info: &vk::PresentInfoKHR) -> VkResult<bool> {
        let queue_guard = self.storage().handle.lock().unwrap();

        let res = unsafe {
            self.device()
                .storage()
                .khr_swapchain
                .queue_present(*queue_guard, info)
        };

        drop(queue_guard);

        res
    }

    #[tracing::instrument(skip_all)]
    pub unsafe fn submit2(
        &self,
        submits: &[vk::SubmitInfo2],
        fence: Option<vk::Fence>,
    ) -> VkResult<SubmissionId> {
        let timeline_value = self
            .storage()
            .timeline_value
            .fetch_add(1, Ordering::Relaxed);

        let queue_guard = self.storage().handle.lock().unwrap();

        let res = unsafe {
            self.device().storage().ash.queue_submit2(
                *queue_guard,
                submits,
                fence.unwrap_or(vk::Fence::null()),
            )
        };

        drop(queue_guard);

        res.map(|()| SubmissionId {
            queue: self.clone(),
            timeline_value,
        })
    }

    pub unsafe fn wait_idle(&self) {
        let queue_guard = self.storage().handle.lock().unwrap();

        unsafe {
            self.device()
                .queue_wait_idle(*queue_guard)
                .expect("wait_idle failed")
        };
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubmissionId {
    pub(crate) queue: Queue,
    pub(crate) timeline_value: u64,
}

/// Unique ID identifying the owner of a resource.
///
/// This is used in cold paths for verifying ownership of a resource. Current users include graphs
/// and upload/download schedulers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OwnerId(u16);

static NEXT_OWNER: AtomicU32 = AtomicU32::new(1);

impl OwnerId {
    pub const NONE: OwnerId = OwnerId(0);

    #[inline]
    pub fn new() -> Self {
        // It doesn't matter what order the IDs are acquired in, only that they are unique. Thus
        // `Ordering::Relaxed` is sufficient.
        let value: u16 = NEXT_OWNER
            .fetch_add(1, Ordering::Relaxed)
            .try_into()
            .expect("OwnerIDs exhausted");

        OwnerId(value)
    }
}

#[derive(Clone)]
pub(crate) struct Ownership<T> {
    owner: Option<OwnerId>,
    state: T,
}

impl<T: Clone> Ownership<T> {
    #[inline]
    pub(crate) fn new(state: T) -> Self {
        Ownership { owner: None, state }
    }

    #[inline]
    pub(crate) fn owner(&self) -> Option<OwnerId> {
        self.owner
    }

    pub(crate) fn acquire(&mut self, owner: OwnerId) -> &T {
        match self.owner {
            // TODO: error
            Some(o) => assert_eq!(o, owner),
            None => self.owner = Some(owner),
        }

        &self.state
    }

    pub(crate) fn release(&mut self, owner: OwnerId, state: Option<T>) {
        match self.owner.take() {
            Some(o) => assert_eq!(o, owner),
            None => panic!("im not owned! im not owned!!"),
        }

        if let Some(state) = state {
            self.state = state;
        }
    }
}

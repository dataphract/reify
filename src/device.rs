use std::{
    ffi::{c_char, CStr},
    sync::{Arc, Mutex},
};

use ash::{ext, khr, prelude::*, vk};
use gpu_allocator::{
    vulkan::{
        Allocation as GpuAllocation, AllocationCreateDesc as GpuAllocationCreateDesc,
        Allocator as GpuAllocator,
    },
    AllocatorDebugSettings as GpuAllocatorDebugSettings,
};
use tracing_log::log;

// Graphics, compute, transfer, present
const _MAX_DEVICE_QUEUES: usize = 4;

pub struct _PhysicalDeviceProperties {
    // From `vk::PhysicalDeviceMaintenance3Properties`
    pub max_per_set_descriptors: u32,
    pub max_memory_allocation_size: u64,
    // From `vk::PhysicalDeviceMaintenance4Properties`
    pub max_buffer_size: u64,
}

#[derive(Clone)]
pub struct PhysicalDevice {
    // Physical device handles do not need to be externally synchronized.
    inner: Arc<PhysicalDeviceInner>,
}

struct PhysicalDeviceInner {
    raw: vk::PhysicalDevice,
    qfp: Vec<vk::QueueFamilyProperties>,
    device_layers: Vec<vk::LayerProperties>,
    device_extensions: Vec<vk::ExtensionProperties>,
}

impl PhysicalDevice {
    // TODO(dp): remove when phys. device enumeration is implemented
    #[allow(clippy::new_without_default)]
    pub fn new() -> PhysicalDevice {
        let instance = crate::instance();

        let phys_devices = unsafe { instance.instance().enumerate_physical_devices().unwrap() };
        let phys_device = phys_devices.into_iter().next().unwrap();

        // SAFETY: No external synchronization requirement.
        let qfp = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(phys_device)
        };

        // SAFETY: No external synchronization requirement.
        let device_layers = unsafe {
            instance
                .instance()
                .enumerate_device_layer_properties(phys_device)
                .unwrap()
        };

        // SAFETY: No external synchronization requirement.
        let device_extensions = unsafe {
            instance
                .instance()
                .enumerate_device_extension_properties(phys_device)
                .unwrap()
        };

        PhysicalDevice {
            inner: Arc::new(PhysicalDeviceInner {
                raw: phys_device,
                qfp,
                device_layers,
                device_extensions,
            }),
        }
    }

    pub fn create_device(&self) -> Device {
        let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;

        let qfp = &self.inner.qfp;
        let (qf_index, _qf) = qfp
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(flags))
            .expect("no queue with GRAPHICS | COMPUTE | TRANSFER capability");
        let qf_index = u32::try_from(qf_index).unwrap();

        // Verify that all necessary device layers are supported.
        //
        // TODO(dp): this is quadratic in the number of layer names
        let required_layer_names: Vec<&CStr> = vec![
            #[cfg(debug_assertions)]
            {
                use crate::instance::LAYER_NAME_VALIDATION;
                LAYER_NAME_VALIDATION
            },
        ];
        for &layer in required_layer_names.iter() {
            assert!(self
                .inner
                .device_layers
                .iter()
                .any(|l| l.layer_name_as_c_str().unwrap() == layer));
        }
        let enabled_layer_names: Vec<*const c_char> =
            required_layer_names.iter().map(|c| c.as_ptr()).collect();

        // Verify that all necessary device extensions are supported.
        //
        // TODO(dp): this is quadratic in the number of extension names
        let required_extension_names = [khr::swapchain::NAME];
        for ext in required_extension_names {
            assert!(self
                .inner
                .device_extensions
                .iter()
                .any(|x| x.extension_name_as_c_str().unwrap() == ext));
        }
        let enabled_extension_names = required_extension_names.map(|c| c.as_ptr());

        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_index)
            .queue_priorities(&[1.0])];

        let mut phys_device_features = vk::PhysicalDeviceFeatures2::default();
        let mut phys_device_features_1_1 = vk::PhysicalDeviceVulkan11Features::default();
        let mut phys_device_features_1_2 =
            vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut phys_device_features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        // Re: enabled_layer_names, per the spec, §46.3.1, "Device Layer Deprecation":
        // > In order to maintain compatibility with implementations released prior to device-layer
        // > deprecation, applications *should* still enumerate and enable device layers.
        #[allow(deprecated)]
        let device_create_info = vk::DeviceCreateInfo::default()
            .flags(vk::DeviceCreateFlags::empty())
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&enabled_layer_names)
            .enabled_extension_names(&enabled_extension_names)
            .push_next(&mut phys_device_features)
            .push_next(&mut phys_device_features_1_1)
            .push_next(&mut phys_device_features_1_2)
            .push_next(&mut phys_device_features_1_3);

        // Safety: no external synchronization requirement.
        let device = unsafe {
            crate::instance()
                .instance()
                .create_device(self.inner.raw, &device_create_info, None)
                .expect("failed to create logical device")
        };

        log::info!("Created logical device.");

        let queue = unsafe { device.get_device_queue(qf_index, 0) };

        let khr_swapchain = khr::swapchain::Device::new(crate::instance().instance(), &device);
        let ext_debug_utils = ext::debug_utils::Device::new(crate::instance().instance(), &device);

        // NOTE: Storing the instance and device inline makes this struct gigantic, nearly 1.5KiB.
        // This shouldn't be large enough to overflow the stack, but it's worth keeping in mind.
        let allocator_info = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: crate::instance().instance().clone(),
            device: device.clone(),
            physical_device: self.raw(),
            debug_settings: GpuAllocatorDebugSettings {
                log_memory_information: true,
                log_leaks_on_shutdown: true,
                store_stack_traces: cfg!(debug_assertions),
                log_allocations: false,
                log_frees: false,
                log_stack_traces: cfg!(debug_assertions),
            },
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };

        let allocator = GpuAllocator::new(&allocator_info).unwrap();

        Device {
            inner: Arc::new(DeviceInner {
                phys_device: self.clone(),
                qf_index,
                queue: Mutex::new(queue),
                raw: device,
                khr_swapchain,
                ext_debug_utils,
                allocator: Mutex::new(allocator),
            }),
        }
    }

    pub fn raw(&self) -> vk::PhysicalDevice {
        self.inner.raw
    }
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    phys_device: PhysicalDevice,
    // Queue family index.
    //
    // Currently, a single queue is used for all operations.
    qf_index: u32,

    // TODO(dp): vk::Queue is Copy, so this can be trivially copied out of the Mutex. Wrap it in
    // something safer.
    queue: Mutex<vk::Queue>,

    raw: ash::Device,
    khr_swapchain: khr::swapchain::Device,
    ext_debug_utils: ext::debug_utils::Device,

    allocator: Mutex<gpu_allocator::vulkan::Allocator>,
}

impl Device {
    pub fn allocate(&self, desc: &GpuAllocationCreateDesc) -> gpu_allocator::Result<GpuAllocation> {
        self.inner.allocator.lock().unwrap().allocate(desc)
    }

    #[allow(clippy::missing_safety_doc)]
    #[tracing::instrument(skip_all)]
    pub unsafe fn acquire_next_image_2(
        &self,
        acquire_info: &vk::AcquireNextImageInfoKHR,
    ) -> VkResult<(u32, bool)> {
        self.inner.khr_swapchain.acquire_next_image2(acquire_info)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn cmd_begin_debug_utils_label(
        &self,
        command_buffer: vk::CommandBuffer,
        label_info: &vk::DebugUtilsLabelEXT,
    ) {
        self.inner
            .ext_debug_utils
            .cmd_begin_debug_utils_label(command_buffer, label_info)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn cmd_end_debug_utils_label(&self, command_buffer: vk::CommandBuffer) {
        self.inner
            .ext_debug_utils
            .cmd_end_debug_utils_label(command_buffer)
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
    pub unsafe fn create_swapchain(
        &self,
        info: &vk::SwapchainCreateInfoKHR,
    ) -> VkResult<vk::SwapchainKHR> {
        self.inner.khr_swapchain.create_swapchain(info, None)
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn get_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> VkResult<Vec<vk::Image>> {
        unsafe { self.inner.khr_swapchain.get_swapchain_images(swapchain) }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn set_debug_utils_object_name(
        &self,
        name_info: &vk::DebugUtilsObjectNameInfoEXT,
    ) -> VkResult<()> {
        unsafe {
            self.inner
                .ext_debug_utils
                .set_debug_utils_object_name(name_info)
        }
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.inner.phys_device
    }

    // TODO(dp): remove this method when dedicated queues are supported
    pub fn queue_family_index(&self) -> u32 {
        self.inner.qf_index
    }

    pub fn transfer_queue_family_index(&self) -> u32 {
        self.inner.qf_index
    }

    #[inline]
    pub fn queue(&self) -> Queue {
        Queue { device: self }
    }

    #[inline]
    pub fn transfer_queue(&self) -> Queue {
        self.queue()
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
        pub unsafe fn create_buffer(info: &vk::BufferCreateInfo) -> VkResult<vk::Buffer>;
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
                    unsafe { self.inner.$name($($param),*) }
                }
            )*
        }

        impl DeviceInner {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { self.raw.$name($($param),*) }
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
                    unsafe { self.inner.$name($($param),*) }
                }
            )*
        }

        impl DeviceInner {
            $(
                $(#[$m])*
                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                $v unsafe fn $name(&self, $($param: $param_ty),*) $(-> $ret_ty)? {
                    // SAFETY: upheld by outer contract.
                    unsafe { self.raw.$name($($param,)* None) }
                }
            )*
        }
    };
}
use device_delegate_no_alloc_callbacks;

pub struct Queue<'device> {
    device: &'device Device,
}

impl<'device> Queue<'device> {
    pub unsafe fn begin_debug_utils_label(&self, label_info: &vk::DebugUtilsLabelEXT) {
        let queue_guard = self.device.inner.queue.lock().unwrap();

        unsafe {
            self.device
                .inner
                .ext_debug_utils
                .queue_begin_debug_utils_label(*queue_guard, label_info)
        };

        drop(queue_guard);
    }

    pub unsafe fn end_debug_utils_label(&self) {
        let queue_guard = self.device.inner.queue.lock().unwrap();

        unsafe {
            self.device
                .inner
                .ext_debug_utils
                .queue_end_debug_utils_label(*queue_guard)
        };

        drop(queue_guard);
    }

    #[tracing::instrument(skip_all)]
    pub unsafe fn present(&self, info: &vk::PresentInfoKHR) -> VkResult<bool> {
        let queue_guard = self.device.inner.queue.lock().unwrap();

        let res = unsafe {
            self.device
                .inner
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
    ) -> VkResult<()> {
        let queue_guard = self.device.inner.queue.lock().unwrap();

        let res = unsafe {
            self.device.inner.raw.queue_submit2(
                *queue_guard,
                submits,
                fence.unwrap_or(vk::Fence::null()),
            )
        };

        drop(queue_guard);

        res
    }
}

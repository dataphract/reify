use std::sync::Arc;

use ash::{khr, prelude::*, vk};
use tracing_log::log;

use crate::instance::LAYER_NAME_VALIDATION;

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
        let required_layer_names = [LAYER_NAME_VALIDATION];
        for layer in required_layer_names {
            assert!(self
                .inner
                .device_layers
                .iter()
                .any(|l| l.layer_name_as_c_str().unwrap() == layer));
        }
        let enabled_layer_names = required_layer_names.map(|c| c.as_ptr());

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

        let khr_swapchain = khr::swapchain::Device::new(crate::instance().instance(), &device);

        Device {
            inner: Arc::new(DeviceInner {
                phys_device: self.clone(),
                qf_index,
                raw: device,
                khr_swapchain,
            }),
        }
    }

    pub fn raw(&self) -> vk::PhysicalDevice {
        self.inner.raw
    }
}

pub struct Device {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    phys_device: PhysicalDevice,
    // Queue family index.
    //
    // Currently, a single queue is used for all operations.
    qf_index: u32,

    raw: ash::Device,
    khr_swapchain: khr::swapchain::Device,
}

impl Device {
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.inner.phys_device
    }

    // TODO(dp): remove this method when dedicated queues are supported
    pub fn queue_family_index(&self) -> u32 {
        self.inner.qf_index
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
} // NOTE: Only add methods here if there is no external synchronization requirement on the device.
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

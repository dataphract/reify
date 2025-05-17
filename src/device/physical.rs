use std::{
    ffi::{c_char, CStr},
    sync::{Arc, Mutex, RwLock},
};

use ash::{ext, khr, vk};
use gpu_allocator::{
    vulkan::Allocator as GpuAllocator, AllocatorDebugSettings as GpuAllocatorDebugSettings,
};
use vk_mem as vma;

use crate::{
    buffer::BufferPool,
    device::{
        extensions::DeviceExtensionFlags, Device, DeviceStorage, Queue, QueueFamily, QueueStorage,
        DEVICE,
    },
};

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
    extensions: DeviceExtensionFlags,
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

        let extensions = DeviceExtensionFlags::detect(&device_extensions);

        PhysicalDevice {
            inner: Arc::new(PhysicalDeviceInner {
                raw: phys_device,
                qfp,
                device_layers,
                extensions,
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
        let required_extensions = DeviceExtensionFlags::KHR_SWAPCHAIN;

        let missing_extensions = required_extensions.difference(self.inner.extensions);

        for name in missing_extensions.iter_ext_names() {
            tracing::error!("missing required extension {}", name.to_str().unwrap());
        }

        let wanted_extensions = DeviceExtensionFlags::EXT_SWAPCHAIN_MAINTENANCE1;

        let enabled_extensions = (required_extensions | wanted_extensions) & self.inner.extensions;

        for ext in enabled_extensions.iter_ext_names() {
            tracing::info!("enabling device extension {}", ext.to_str().unwrap());
        }

        let enabled_extension_names = enabled_extensions
            .iter_ext_names()
            .map(|c| c.as_ptr())
            .collect::<Vec<_>>();

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

        tracing::info!("Created logical device.");

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

        let single_queue = Queue {
            family: QueueFamily { index: qf_index },
            index: 0,
        };

        // TODO(dp): replace with dedicated queues
        let graphics_queue = single_queue.clone();
        let upload_queue = single_queue.clone();
        let download_queue = single_queue.clone();
        let queues = vec![unsafe { QueueStorage::create(&device, queue) }];

        // TODO: custom capacity
        let buffers = RwLock::new(BufferPool::with_capacity(65536));

        let res = DEVICE.set(DeviceStorage {
            phys_device: self.clone(),
            graphics_queue,
            upload_queue,
            _download_queue: download_queue,
            queues,
            ash: device,
            khr_swapchain,
            ext_debug_utils,
            allocator: Mutex::new(allocator),
            buffers,
        });

        assert!(res.is_ok());

        Device {}
    }

    pub fn raw(&self) -> vk::PhysicalDevice {
        self.inner.raw
    }
}

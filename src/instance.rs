use std::{cmp, ffi::CStr, sync::OnceLock};

use ash::{ext, khr, prelude::VkResult, vk};
use tracing_log::log;

//use crate::debug::DebugMessenger;

pub(crate) const LAYER_NAME_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

static INSTANCE: OnceLock<Instance> = OnceLock::new();
// static DEBUG_MESSENGER: OnceLock<DebugMessenger> = OnceLock::new();

fn required_extensions() -> VkResult<Vec<&'static CStr>> {
    let entry = crate::entry();
    let instance_version = unsafe {
        entry
            .try_enumerate_instance_version()?
            .unwrap_or(vk::API_VERSION_1_0)
    };

    let version_str = format!(
        "{}.{}.{}",
        vk::api_version_major(instance_version),
        vk::api_version_minor(instance_version),
        vk::api_version_patch(instance_version),
    );

    // TODO: do more compat work for lower instance versions.
    if instance_version < vk::API_VERSION_1_3 {
        log::error!(
            "Vulkan instance version too low: must be at least 1.3.0, but is {version_str}"
        );
        panic!("Incompatible Vulkan instance version");
    }

    log::info!("Vulkan instance version: {version_str}");
    let instance_extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };

    let mut extensions = Vec::new();

    extensions.push(khr::surface::NAME);
    if cfg!(all(
        unix,
        not(target_os = "android"),
        not(target_os = "macos")
    )) {
        extensions.push(khr::wayland_surface::NAME);
        extensions.push(khr::xcb_surface::NAME);
        extensions.push(khr::xlib_surface::NAME);
    } else {
        unimplemented!("only tested on linux at the moment, sorry :(");
    }

    extensions.push(ext::debug_utils::NAME);
    extensions.push(khr::display::NAME);

    extensions.retain(|&wanted| {
        for inst_ext in instance_extensions.iter() {
            let name_bytes: &[u8] = bytemuck::cast_slice(&inst_ext.extension_name);
            let name = CStr::from_bytes_until_nul(name_bytes).unwrap();
            if wanted == name {
                return true;
            }
        }

        log::warn!("Extension not found: {}", wanted.to_string_lossy());
        false
    });

    Ok(extensions)
}

/// Lists the set of required layers.
fn required_layers() -> VkResult<Vec<&'static CStr>> {
    let entry = crate::entry();
    let instance_layers = unsafe { entry.enumerate_instance_layer_properties()? };

    let mut layers = Vec::new();
    if cfg!(debug_assertions) {
        layers.push(LAYER_NAME_VALIDATION);
    }

    layers.retain(|&wanted| {
        for inst_layer in instance_layers.iter() {
            let name_bytes: &[u8] = bytemuck::cast_slice(&inst_layer.layer_name);
            let name = CStr::from_bytes_until_nul(name_bytes).unwrap();
            if wanted == name {
                return true;
            }
        }

        log::warn!("Layer not found: {}", wanted.to_string_lossy());
        false
    });

    Ok(layers)
}

pub struct Instance {
    instance: ash::Instance,
    khr_surface: khr::surface::Instance,

    extensions: Vec<&'static CStr>,
    _layers: Vec<&'static CStr>,
}

impl Instance {
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn khr_surface(&self) -> &khr::surface::Instance {
        &self.khr_surface
    }

    pub fn has_wayland(&self) -> bool {
        self.extensions.contains(&khr::wayland_surface::NAME)
    }
}

pub fn instance() -> &'static Instance {
    // TODO(dp): use get_or_try_init when stabilized
    INSTANCE
        .get_or_init(|| -> Instance { create_instance().expect("Vulkan instance creation failed") })
}

fn create_instance() -> VkResult<Instance> {
    let entry = crate::entry();
    let driver_api_version = unsafe {
        entry
            .try_enumerate_instance_version()?
            .unwrap_or(vk::API_VERSION_1_0)
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"reify")
        .application_version(0)
        .engine_name(c"reify")
        .api_version(cmp::min(vk::API_VERSION_1_3, driver_api_version));

    let extensions = required_extensions()?;
    for ext in &extensions {
        log::info!("enabling extension {}", ext.to_string_lossy())
    }
    let ext_ptrs = extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    let layers = required_layers()?;
    for layer in &layers {
        log::info!("enabling layer {}", layer.to_string_lossy())
    }
    let layer_ptrs = layers.iter().map(|lyr| lyr.as_ptr()).collect::<Vec<_>>();

    let create_info = vk::InstanceCreateInfo::default()
        .flags(vk::InstanceCreateFlags::empty())
        .application_info(&app_info)
        .enabled_extension_names(&ext_ptrs)
        .enabled_layer_names(&layer_ptrs);

    let instance = unsafe { entry.create_instance(&create_info, None)? };

    // DEBUG_MESSENGER
    // .get_or_init(|| unsafe { DebugMessenger::new(DebugUtils::new(entry, &instance)) });

    Ok(Instance {
        khr_surface: khr::surface::Instance::new(entry, &instance),
        instance,
        extensions,
        _layers: layers,
    })
}

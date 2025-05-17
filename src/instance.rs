use std::{
    cmp,
    ffi::CStr,
    fmt::{self, Write},
    sync::OnceLock,
};

use ash::{ext, khr, prelude::VkResult, vk};

use crate::device::extensions::InstanceExtensionFlags;

pub(crate) const LAYER_NAME_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

static INSTANCE: OnceLock<Instance> = OnceLock::new();
static DEBUG_MESSENGER: OnceLock<DebugMessenger> = OnceLock::new();

fn required_extensions() -> VkResult<InstanceExtensionFlags> {
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
    let extension_props = unsafe { entry.enumerate_instance_extension_properties(None)? };
    let available = InstanceExtensionFlags::detect(&extension_props);

    let mut wanted = InstanceExtensionFlags::KHR_DISPLAY
        | InstanceExtensionFlags::KHR_SURFACE
        | InstanceExtensionFlags::EXT_DEBUG_UTILS;

    if cfg!(all(
        unix,
        not(target_os = "android"),
        not(target_os = "macos")
    )) {
        wanted |= InstanceExtensionFlags::KHR_WAYLAND_SURFACE
            | InstanceExtensionFlags::KHR_XCB_SURFACE
            | InstanceExtensionFlags::KHR_XLIB_SURFACE;
    } else {
        unimplemented!("only tested on linux at the moment, sorry :(");
    }

    let missing = wanted.difference(available);
    for flag in missing.iter_ext_names() {
        log::error!("missing required extension: {:?}", flag);
    }

    Ok(wanted)
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
    api_version: u32,

    instance: ash::Instance,
    khr_surface: khr::surface::Instance,

    extensions: InstanceExtensionFlags,
    _layers: Vec<&'static CStr>,
}

impl Instance {
    #[inline]
    pub fn version(&self) -> u32 {
        self.api_version
    }

    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    #[inline]
    pub fn khr_surface(&self) -> &khr::surface::Instance {
        &self.khr_surface
    }

    pub fn has_wayland(&self) -> bool {
        self.extensions
            .contains(InstanceExtensionFlags::KHR_WAYLAND_SURFACE)
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

    let api_version = cmp::min(vk::API_VERSION_1_3, driver_api_version);

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"reify")
        .application_version(0)
        .engine_name(c"reify")
        .api_version(api_version);

    let extensions = required_extensions()?;
    for ext in extensions.iter_ext_names() {
        log::info!("enabling extension {}", ext.to_string_lossy())
    }
    let ext_ptrs = extensions
        .iter_ext_names()
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

    DEBUG_MESSENGER.get_or_init(|| unsafe {
        DebugMessenger::new(ext::debug_utils::Instance::new(entry, &instance))
    });

    Ok(Instance {
        api_version,
        khr_surface: khr::surface::Instance::new(entry, &instance),
        instance,
        extensions,
        _layers: layers,
    })
}

fn format_cstr<F, C>(f: &mut F, cstr: C) -> fmt::Result
where
    F: fmt::Write,
    C: AsRef<CStr>,
{
    let mut start = 0;
    let bytes = cstr.as_ref().to_bytes();

    'format: while start < bytes.len() {
        let unvalidated = &bytes[start..];

        match std::str::from_utf8(unvalidated) {
            Ok(s) => {
                f.write_str(s)?;
                start += s.len();
            }

            Err(e) => {
                // Safety: validated up to `e.valid_up_to()`.
                let valid =
                    unsafe { std::str::from_utf8_unchecked(&unvalidated[..e.valid_up_to()]) };

                f.write_str(valid)?;
                f.write_char(char::REPLACEMENT_CHARACTER)?;
                match e.error_len() {
                    // Skip the validated substring and the unrecognized sequence.
                    Some(l) => start += valid.len() + l,

                    // Unexpected end of input.
                    None => break 'format,
                }
            }
        }
    }

    Ok(())
}

/// Format a list of `vk::DebugUtilsLabelEXT` objects.
///
/// # Safety
///
/// `labels` must be a pointer to a properly aligned sequence of `count`
/// `vk::DebugUtilsLabelEXT` objects.
unsafe fn format_debug_utils_label_ext<F>(
    f: &mut F,
    about: &str,
    labels: *mut vk::DebugUtilsLabelEXT,
    count: usize,
) -> fmt::Result
where
    F: fmt::Write,
{
    if labels.is_null() {
        return Ok(());
    }

    let labels = unsafe { std::slice::from_raw_parts(labels, count) };

    let (last, init) = match labels.split_last() {
        Some(li) => li,
        None => return Ok(()),
    };

    write!(f, "{}: ", about)?;

    for label in init {
        if let Some(label_ptr) = unsafe { label.p_label_name.as_ref() } {
            let label_cstr = unsafe { CStr::from_ptr(label_ptr) };
            format_cstr(f, label_cstr)?;
            f.write_str(", ")?;
        }
    }

    if let Some(label_ptr) = unsafe { last.p_label_name.as_ref() } {
        let label_cstr = unsafe { CStr::from_ptr(label_ptr) };
        format_cstr(f, label_cstr)?;
    }

    Ok(())
}

/// Format a list of `vk::DebugUtilsObjectNameInfoEXT` objects.
///
/// # Safety
///
/// `infos` must be a pointer to a properly aligned sequence of `count`
/// `vk::DebugUtilsObjectNameInfoEXT` objects.
unsafe fn format_debug_utils_object_name_info_ext<F>(
    f: &mut F,
    about: &str,
    infos: *mut vk::DebugUtilsObjectNameInfoEXT,
    count: usize,
) -> fmt::Result
where
    F: fmt::Write,
{
    if infos.is_null() {
        return Ok(());
    }

    let infos = unsafe { std::slice::from_raw_parts(infos, count) };

    let (last, init) = match infos.split_last() {
        Some(li) => li,
        None => return Ok(()),
    };

    for info in init {
        write!(
            f,
            "{}: (type: {:?}, handle: 0x{:X}",
            about, info.object_type, info.object_handle
        )?;
        if let Some(info_ptr) = unsafe { info.p_object_name.as_ref() } {
            let info_cstr = unsafe { CStr::from_ptr(info_ptr) };
            f.write_str("name: ")?;
            format_cstr(f, info_cstr)?;
        }
        f.write_str("), ")?;
    }

    write!(
        f,
        "{}: (type: {:?}, handle: 0x{:X}",
        about, last.object_type, last.object_handle
    )?;
    if let Some(info_ptr) = unsafe { last.p_object_name.as_ref() } {
        let info_cstr = unsafe { CStr::from_ptr(info_ptr) };
        f.write_str("name: ")?;
        format_cstr(f, info_cstr)?;
    }
    f.write_str(")")?;

    Ok(())
}
const DEBUG_MESSAGE_INIT_CAPACITY: usize = 128;
unsafe extern "system" fn debug_utils_messenger_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // Via the spec: "The application should always return VK_FALSE."

    if std::thread::panicking() {
        return vk::FALSE;
    }

    if let Err(e) = debug_utils_messenger_callback_impl(severity, ty, callback_data, user_data) {
        log::error!("debug message formatting failed: {}", e);
    }

    vk::FALSE
}

fn debug_utils_messenger_callback_impl(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> fmt::Result {
    if callback_data.is_null() {
        return Ok(());
    }

    let callback_data = unsafe { *callback_data };

    let severity = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Trace,
        _ => log::Level::Warn,
    };

    let mut log_message = String::with_capacity(DEBUG_MESSAGE_INIT_CAPACITY);

    write!(&mut log_message, "{:?} ", ty)?;

    let msg_id_num = callback_data.message_id_number;
    if callback_data.p_message_id_name.is_null() {
        // "[Message MESSAGE_ID]"
        write!(&mut log_message, "[Message 0x{:X}] : ", msg_id_num)?;
    } else {
        // "[MESSAGE_NAME (MESSAGE_ID)]"
        let msg_name_cstr = unsafe { CStr::from_ptr(callback_data.p_message_id_name) };
        log_message.write_str("[")?;
        format_cstr(&mut log_message, msg_name_cstr)?;
        write!(&mut log_message, " (0x{:X})] : ", msg_id_num)?;
    };

    if !callback_data.p_message.is_null() {
        let msg_cstr = unsafe { CStr::from_ptr(callback_data.p_message) };
        format_cstr(&mut log_message, msg_cstr)?;
    };

    unsafe {
        format_debug_utils_label_ext(
            &mut log_message,
            "queue info",
            callback_data.p_queue_labels as *mut _,
            callback_data.queue_label_count as usize,
        )?;

        format_debug_utils_label_ext(
            &mut log_message,
            "cmdbuf info",
            callback_data.p_cmd_buf_labels as *mut _,
            callback_data.cmd_buf_label_count as usize,
        )?;

        format_debug_utils_object_name_info_ext(
            &mut log_message,
            "queue info",
            callback_data.p_queue_labels as *mut _,
            callback_data.queue_label_count as usize,
        )?;
    }

    log::log!(severity, "{}", log_message);

    Ok(())
}

pub struct DebugMessenger {
    _messenger: vk::DebugUtilsMessengerEXT,
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        // TODO
        log::warn!("Debug messenger not destroyed!");
    }
}

impl DebugMessenger {
    /// Initializes a new `DebugUtils`.
    pub unsafe fn new(utils: ext::debug_utils::Instance) -> DebugMessenger {
        let debug_ext_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                // TODO
                // | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING,
            )
            .pfn_user_callback(Some(debug_utils_messenger_callback));

        let messenger = unsafe { utils.create_debug_utils_messenger(&debug_ext_info, None) }
            .expect("failed to create debug messenger");

        utils.submit_debug_utils_message(
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            &vk::DebugUtilsMessengerCallbackDataEXT::default()
                .message(c"DebugMessenger initialized"),
        );

        DebugMessenger {
            _messenger: messenger,
        }
    }
}

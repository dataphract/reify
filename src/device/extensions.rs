// Faster Vulkan extension lookup.
//
// The device exposes its extensions by filling an array of property structs with fixed-size string
// fields. This is fine for initial detection but we can't be doing string comparisons at runtime.
// Instead, we detect all the features we're interested in at startup and stick them in a set of
// flags.
//
// Note that these flag values don't correspond to Vulkan's extension numbers -- we won't need most
// extensions, and by keeping the names in sorted order we can binary search through them during
// detection.

declare_extensions! {
    pub struct InstanceExtensionFlags(InstanceExtensionValues): u64 {
        const EXT_DEBUG_UTILS = ext::debug_utils;
        const KHR_DISPLAY = khr::display;
        const KHR_SURFACE = khr::surface;
        const KHR_WAYLAND_SURFACE = khr::wayland_surface;
        const KHR_XCB_SURFACE = khr::xcb_surface;
        const KHR_XLIB_SURFACE = khr::xlib_surface;
    }
}

declare_extensions! {
    pub struct DeviceExtensionFlags(ExtensionValues): u64 {
        const KHR_DEDICATED_ALLOCATION = khr::dedicated_allocation;
        const KHR_SWAPCHAIN = khr::swapchain;
    }
}

macro_rules! declare_extensions {
    (
        $v:vis struct $name:ident($values:ident): $bits:ident {
            $(
                const $variant:ident = $vendor:ident :: $extension:ident;
            )*
        }
    ) => {
        #[allow(non_camel_case_types)]
        enum $values {
            $($variant,)*
        }

        bitflags::bitflags! {
            #[derive(Copy, Clone, PartialEq, Eq)]
            $v struct $name: $bits {
                $(
                    const $variant = 1 << ($values::$variant as u32);
                )*
            }
        }

        impl $name {
            // Sorted list of names.
            const NAMES: &[&std::ffi::CStr] = {
                &[$(ash::$vendor::$extension::NAME,)*]
            };

            // Make sure the list is actually sorted.
            const TESTS: () = {
                #[cfg(test)]
                mod tests {
                    use super::$name;

                    #[test]
                    fn names_sorted() {
                        assert!($name::NAMES.is_sorted());
                    }
                }
            };


            pub fn detect(extensions: &[vk::ExtensionProperties]) -> Self {
                let mut flags = Self::empty();

                for props in extensions {
                    let Ok(name) = props.extension_name_as_c_str() else {
                        continue;
                    };

                    let Some(ext) = Self::from_ext_name(name) else {
                        continue;
                    };

                    flags |= ext;
                }

                flags
            }

            fn from_ext_name(name: &std::ffi::CStr) -> Option<Self> {
                let idx = Self::NAMES.binary_search(&name).ok()?;

                match idx {
                    $(x if $values::$variant as usize == x => Some(Self::$variant),)*
                    _ => None,
                }
            }

            pub fn iter_ext_names(self) -> impl Iterator<Item = &'static std::ffi::CStr> {
                (0..$bits::BITS)
                    .take(Self::NAMES.len())
                    .filter_map(move |shift| {
                        (self.bits() & (1 << shift) != 0).then_some(Self::NAMES[shift as usize])
                    })
            }
        }
    };
}

use ash::vk;
use declare_extensions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_sorted() {
        assert!(DeviceExtensionFlags::NAMES.is_sorted());
        assert!(InstanceExtensionFlags::NAMES.is_sorted());
    }
}

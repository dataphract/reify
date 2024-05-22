use std::{cmp, marker::PhantomData};

use ash::vk;
use tracing_log::log;

use crate::Device;

const _MAX_FRAMES_IN_FLIGHT: usize = 2;

/// A swapchain image, along with the resources used to render to it.
pub struct SwapchainImage {
    pub(crate) _index: u32,
    pub(crate) _view: vk::ImageView,
    pub(crate) _image: vk::Image,

    unsync: PhantomUnSync,
    unsend: PhantomUnSend,
}
pub struct DisplayInfo {
    pub min_image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

pub struct Display {
    _info: DisplayInfo,

    _current_frame: u64,

    //frames: Vec<FrameContextState>,
    _images: Vec<SwapchainImage>,
    _image_frames: Vec<Option<usize>>,

    _swapchain: Option<vk::SwapchainKHR>,
    _surface: Option<vk::SurfaceKHR>,
}

impl Display {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn create(
        device: &Device,
        surface: vk::SurfaceKHR,
        phys_window_extent: vk::Extent2D,
    ) -> Display {
        let instance = crate::instance();

        let (surf_caps, surf_formats, surf_present_modes) = unsafe {
            let phys = device.physical_device().raw();

            let surface_supported = instance
                .khr_surface()
                .get_physical_device_surface_support(phys, device.queue_family_index(), surface)
                .expect("failed to verify physical device surface support");

            if !surface_supported {
                panic!("Surface not supported with this device.");
            }

            (
                instance
                    .khr_surface()
                    .get_physical_device_surface_capabilities(phys, surface)
                    .expect("failed to query surface capabilities"),
                instance
                    .khr_surface()
                    .get_physical_device_surface_formats(phys, surface)
                    .expect("failed to query surface formats"),
                instance
                    .khr_surface()
                    .get_physical_device_surface_present_modes(phys, surface)
                    .expect("failed to query surface presentation modes"),
            )
        };

        let min_image_count = {
            // Try to keep an image free from the driver at all times.
            let desired = surf_caps.min_image_count + 1;

            if surf_caps.max_image_count == 0 {
                // No limit.
                desired
            } else {
                cmp::min(desired, surf_caps.max_image_count)
            }
        };

        // This is highly unlikely, but the spec doesn't require that implementations support the
        // identity transform.
        assert!(
            surf_caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY),
            "surface must support IDENTITY_KHR transform",
        );

        // Prefer BGRA sRGB if available.
        let surface_format = *surf_formats
            .iter()
            .find(|sf| {
                sf.format == vk::Format::B8G8R8A8_SRGB
                    && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surf_formats[0]);

        const SWAPCHAIN_CHOOSES_EXTENT: vk::Extent2D = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };

        let image_extent = if surf_caps.current_extent == SWAPCHAIN_CHOOSES_EXTENT {
            phys_window_extent
        } else {
            surf_caps.current_extent
        };

        let present_mode = if surf_present_modes
            .iter()
            .any(|&pm| pm == vk::PresentModeKHR::MAILBOX)
        {
            vk::PresentModeKHR::MAILBOX
        } else {
            // Implementations are required to support FIFO.
            vk::PresentModeKHR::FIFO
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(image_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain =
            unsafe { device.create_swapchain(&create_info) }.expect("failed to create swapchain");

        log::info!("Created swapchain.");

        let images = unsafe { device.get_swapchain_images(swapchain) }
            .expect("failed to get swapchain images");

        log::info!("Retrieved {} images from swapchain.", images.len());

        let view_info = |img| {
            vk::ImageViewCreateInfo::default()
                .flags(vk::ImageViewCreateFlags::empty())
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        };

        let image_views = images
            .iter()
            .map(|&img| unsafe { device.create_image_view(&view_info(img)) })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let swapchain_images = images
            .into_iter()
            .enumerate()
            .zip(image_views)
            .map(|((index, image), view)| SwapchainImage {
                _index: index as u32,
                _view: view,
                _image: image,
                unsend: PhantomData,
                unsync: PhantomData,
            })
            .collect::<Vec<_>>();

        let info = DisplayInfo {
            min_image_count,
            surface_format,
            image_extent,
            present_mode,
        };

        // TODO
        // let mut frames = Vec::new();
        // for _ in 0..MAX_FRAMES_IN_FLIGHT {
        //     frames.push(FrameContextState::create(device));
        // }

        let mut image_frames = Vec::with_capacity(swapchain_images.len());
        image_frames.resize(swapchain_images.len(), None);

        Display {
            _info: info,
            _current_frame: 0,
            // frames,
            _images: swapchain_images,
            _image_frames: image_frames,
            _swapchain: Some(swapchain),
            _surface: Some(surface),
            // context: context.clone(),
        }
    }
}

type PhantomUnSend = PhantomData<UnSend>;
type PhantomUnSync = PhantomData<UnSync>;

struct UnSend {
    _p: *const (),
}
unsafe impl Sync for UnSend {}

struct UnSync {
    _p: *const (),
}
unsafe impl Send for UnSync {}

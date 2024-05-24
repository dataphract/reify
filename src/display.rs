use std::{cmp, marker::PhantomData};

use ash::vk;
use tracing_log::log;

use crate::{
    frame::{FrameContext, FrameResources},
    Device,
};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// A swapchain image, along with the resources used to render to it.
pub struct SwapchainImage {
    pub(crate) index: u32,
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,

    unsync: PhantomUnSync,
    unsend: PhantomUnSend,
}

impl SwapchainImage {
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

pub struct DisplayInfo {
    pub min_image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
}

pub struct Display {
    info: DisplayInfo,

    current_frame: usize,

    frames: Vec<FrameResources>,
    images: Vec<SwapchainImage>,
    _image_frames: Vec<Option<usize>>,

    swapchain: Option<vk::SwapchainKHR>,
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

        let (surf_caps, surf_formats, _surf_present_modes) = unsafe {
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

        // TODO(dp): make configurable
        //
        // Implementations are required to support FIFO.
        let present_mode = vk::PresentModeKHR::FIFO;

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
                index: index as u32,
                view,
                image,
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

        let mut frames = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frames.push(FrameResources::create(device));
        }

        let mut image_frames = Vec::with_capacity(swapchain_images.len());
        image_frames.resize(swapchain_images.len(), None);

        Display {
            info,
            current_frame: 0,
            frames,
            images: swapchain_images,
            _image_frames: image_frames,
            swapchain: Some(swapchain),
            _surface: Some(surface),
            // context: context.clone(),
        }
    }

    pub fn acquire_frame_context<'frame>(
        &'frame mut self,
        device: &Device,
    ) -> FrameContext<'frame> {
        let frame_idx = self.current_frame % MAX_FRAMES_IN_FLIGHT;
        let frame = &mut self.frames[frame_idx];

        let frame_cx = frame.acquire_context(device, None).unwrap();

        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.swapchain.unwrap())
            .timeout(u64::MAX)
            .semaphore(frame_cx.image_available())
            .fence(vk::Fence::null())
            .device_mask(1);

        let (acquired, is_suboptimal) = unsafe { device.acquire_next_image_2(&acquire_info) }
            .expect("failed to acquire next swapchain image");

        if is_suboptimal {
            log::warn!("swapchain image is suboptimal");
        }

        frame_cx.attach(
            device,
            &self.info,
            self.swapchain.unwrap(),
            &mut self.images[acquired as usize],
        )
    }

    pub fn info(&self) -> &DisplayInfo {
        &self.info
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

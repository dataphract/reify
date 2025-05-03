use std::{cmp, marker::PhantomData};

use ash::vk;
use tracing_log::log;

use crate::{
    frame::{FrameContext, FrameResources},
    image::{ImageInfo, ImageTiling},
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
    #[inline]
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    #[inline]
    pub unsafe fn destroy(self, device: &Device) {
        // The vk::Image itself is owned by the swapchain, so only destroy the view.
        unsafe { device.destroy_image_view(self.view) }
    }
}

pub struct DisplayInfo {
    pub min_image_count: u32,
    pub surface_format: vk::SurfaceFormatKHR,
    pub image_info: ImageInfo,
    pub present_mode: vk::PresentModeKHR,
}

pub struct Display {
    info: Option<DisplayInfo>,

    current_frame: usize,

    frames: Vec<FrameResources>,
    images: Vec<SwapchainImage>,
    _image_frames: Vec<Option<usize>>,

    swapchain: Option<vk::SwapchainKHR>,
    surface: vk::SurfaceKHR,
}

impl Display {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn create(
        device: &Device,
        surface: vk::SurfaceKHR,
        phys_window_extent: vk::Extent2D,
    ) -> Display {
        let mut display = Display {
            info: None,
            current_frame: 0,

            frames: vec![],
            images: vec![],
            _image_frames: vec![],

            swapchain: None,
            surface,
        };

        unsafe { display.recreate(device, phys_window_extent) };

        display
    }

    #[tracing::instrument(name = "Display::recreate", skip_all)]
    pub unsafe fn recreate(&mut self, device: &Device, phys_window_extent: vk::Extent2D) {
        let instance = crate::instance();

        for frame in self.frames.drain(..) {
            unsafe { frame.destroy(device) };
        }

        for img in self.images.drain(..) {
            unsafe { img.destroy(device) };
        }

        self._image_frames.clear();

        let surf_caps;
        let surf_formats;
        let _surf_present_modes;

        unsafe {
            let phys = device.physical_device().raw();

            let surface_supported = instance
                .khr_surface()
                .get_physical_device_surface_support(
                    phys,
                    device.graphics_queue().family().as_u32(),
                    self.surface,
                )
                .expect("failed to verify physical device surface support");

            if !surface_supported {
                panic!("Surface not supported with this device.");
            }

            surf_caps = instance
                .khr_surface()
                .get_physical_device_surface_capabilities(phys, self.surface)
                .expect("failed to query surface capabilities");
            surf_formats = instance
                .khr_surface()
                .get_physical_device_surface_formats(phys, self.surface)
                .expect("failed to query surface formats");
            _surf_present_modes = instance
                .khr_surface()
                .get_physical_device_surface_present_modes(phys, self.surface)
                .expect("failed to query surface presentation modes");
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
        //
        // TODO(dp): this doesn't make sense on all platforms
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
            .surface(self.surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(image_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(self.swapchain.unwrap_or_else(vk::SwapchainKHR::null));

        let swapchain =
            unsafe { device.create_swapchain(&create_info) }.expect("failed to create swapchain");
        self.swapchain = Some(swapchain);

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
                .components(vk::ComponentMapping::default())
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

        let swapchain_image_info = ImageInfo {
            format: surface_format.format,
            extent: image_extent.into(),
            tiling: ImageTiling::Optimal,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        };

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

        self.info = Some(DisplayInfo {
            min_image_count,
            surface_format,
            image_info: swapchain_image_info,
            present_mode,
        });

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            self.frames.push(FrameResources::create(device));
        }

        self._image_frames.resize(swapchain_images.len(), None);

        self.images = swapchain_images;
    }

    pub fn acquire_frame_context<'frame>(
        &'frame mut self,
        device: &Device,
    ) -> Result<FrameContext<'frame>, AcquireError<'frame>> {
        let frame_idx = self.current_frame % MAX_FRAMES_IN_FLIGHT;
        let frame = &mut self.frames[frame_idx];

        let frame_cx = frame.acquire_context(device, None).unwrap();

        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.swapchain.unwrap())
            .timeout(u64::MAX)
            .semaphore(frame_cx.image_available())
            .fence(vk::Fence::null())
            .device_mask(1);

        let (acquired, is_suboptimal) = match unsafe { device.acquire_next_image_2(&acquire_info) }
        {
            Ok((a, s)) => (a, s),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(AcquireError::OutOfDate),
            Err(e) => panic!("failed to acquire next swapchain image: {e:?}"),
        };

        let frame_cx = frame_cx.attach(
            device,
            self.info.as_ref().unwrap(),
            self.swapchain.unwrap(),
            &mut self.images[acquired as usize],
        );

        if is_suboptimal {
            return Err(AcquireError::Suboptimal(frame_cx));
        }

        Ok(frame_cx)
    }

    pub fn info(&self) -> &DisplayInfo {
        &self.info.as_ref().unwrap()
    }
}

/// An error that may be returned by [`Display::acquire_frame_context`].
pub enum AcquireError<'frame> {
    /// The swapchain is suboptimal for the current window surface.
    ///
    /// The provided frame context can still be used for rendering, but the application should
    /// recreate the [`Display`].
    Suboptimal(FrameContext<'frame>),
    /// The swapchain is incompatible with the window surface.
    ///
    /// The application must recreate the [`Display`] in order to render.
    OutOfDate,
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

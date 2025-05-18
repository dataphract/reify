// Some notes on Vulkan's swapchain API:
//
// We need to ask the driver which image to use for the next frame. The driver does that by
// returning the index of an image from acquire_next_image(). It also accepts a fence and a
// semaphore which it will signal when the image is actually usable.
//
// The fence and semaphore we use to order our writes after the driver's prior reads can't be
// associated with the swapchain image index because the driver needs to know the fence and
// semaphore to signal at the time of the call, _before_ we know which index to use.
//
// However, the semaphore we use to order our writes before the driver's subsequent reads _does_
// need to be associated with the image index. The spec requires that the semaphore is unsignaled
// when we submit it to be signaled. That means the driver needs to have waited on it already, and
// the driver has no way of telling us which semaphores it's waited on except to give us an
// associated image index.
//
// This is addressed by VK_EXT_swapchain_maintenance1, which finally allows the driver to signal a
// fence when it's done waiting on a semaphore. However, we can't rely on this extension being
// available; at time of writing the coverage is ~60% on Linux and ~40% on Windows.

use std::{cmp, marker::PhantomData};

use ash::{prelude::VkResult, vk};

use crate::{
    frame::{FrameContext, FrameResources},
    image::{ImageInfo, ImageTiling},
    Device,
};

/// A swapchain image, along with the resources used to render to it.
pub struct SwapchainImage {
    pub(crate) index: u32,
    pub(crate) view: vk::ImageView,
    pub(crate) image: vk::Image,
    pub(crate) present_wait: vk::Semaphore,

    unsync: PhantomUnSync,
    unsend: PhantomUnSend,
}

impl SwapchainImage {
    fn create(device: &Device, info: &ImageInfo, image: vk::Image, index: usize) -> VkResult<Self> {
        let view_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(info.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view;
        let present_wait;
        unsafe {
            view = device.create_image_view(&view_info)?;

            let sem_info = vk::SemaphoreCreateInfo::default();
            present_wait = device.create_semaphore(&sem_info)?;

            device.set_debug_utils_object_name_with(
                image,
                format_args!("swapchain.images[{index}].image"),
            )?;
            device.set_debug_utils_object_name_with(
                view,
                format_args!("swapchain.images[{index}].view"),
            )?;
            device.set_debug_utils_object_name_with(
                present_wait,
                format_args!("swapchain.images[{index}].present_wait"),
            )?;
        }

        Ok(SwapchainImage {
            index: index as u32,
            image,
            view,
            present_wait,
            unsend: PhantomData,
            unsync: PhantomData,
        })
    }

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
            // Try to triple buffer if possible.
            let desired = cmp::max(surf_caps.min_image_count, 3);

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

        let supported_present_modes = instance
            .khr_surface()
            .get_physical_device_surface_present_modes(device.physical_device().raw(), self.surface)
            .unwrap();

        // TODO(dp): make configurable
        let present_mode = if supported_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            // Implementations are required to support FIFO.
            vk::PresentModeKHR::FIFO
        };

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

        let swapchain_image_info = ImageInfo {
            format: surface_format.format,
            extent: image_extent.into(),
            tiling: ImageTiling::Optimal,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        };

        let swapchain_images = images
            .into_iter()
            .enumerate()
            .map(|(index, image)| {
                SwapchainImage::create(device, &swapchain_image_info, image, index)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        self.info = Some(DisplayInfo {
            min_image_count,
            surface_format,
            image_info: swapchain_image_info,
            present_mode,
        });

        for _ in 0..swapchain_images.len() {
            self.frames.push(FrameResources::create(device));
        }

        self.images = swapchain_images;
    }

    pub fn acquire_frame_context<'frame>(
        &'frame mut self,
        device: &Device,
    ) -> Result<FrameContext<'frame>, AcquireError<'frame>> {
        self.current_frame += 1;

        let num_frames = self.frames.len();
        let frame = &mut self.frames[self.current_frame % num_frames];
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

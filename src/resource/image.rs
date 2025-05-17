use ash::{prelude::VkResult, vk};
use vk_mem::{self as vma, Alloc as _};

use crate::Device;

pub struct Image {
    pub(crate) info: ImageInfo,
    pub(crate) handle: vk::Image,
    pub(crate) mem: vma::Allocation,
    pub(crate) default_view: vk::ImageView,
}

impl Image {
    pub fn create(device: &Device, info: &ImageInfo) -> VkResult<Self> {
        let create_info: vk::ImageCreateInfo = info.into();

        let alloc_info = vma::AllocationCreateInfo {
            flags: vma::AllocationCreateFlags::empty(),
            usage: vma::MemoryUsage::Auto,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            memory_type_bits: u32::MAX,
            user_data: 0,
            priority: 0.5,
        };

        let (image, mem) = unsafe { device.allocator().create_image(&create_info, &alloc_info)? };

        let view_type = match info.extent {
            ImageExtent::D2(_) => vk::ImageViewType::TYPE_2D,
        };

        let default_aspects = info.format.aspects();
        let view_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(view_type)
            .format(info.format)
            .components(vk::ComponentMapping::default())
            .subresource_range(subresource_range_full(default_aspects));

        let default_view = unsafe { device.create_image_view(&view_info)? };

        Ok(Image {
            info: info.clone(),
            handle: image,
            mem,
            default_view,
        })
    }

    /// Destroys the image and frees its associated memory.
    ///
    /// # Safety
    ///
    /// All accesses to the image memory must be completed before this method is called.
    pub unsafe fn destroy(self, device: &Device) {
        let Image {
            info: _,
            handle,
            mut mem,
            default_view,
        } = self;

        unsafe {
            device.destroy_image_view(default_view);
            device.allocator().destroy_image(handle, &mut mem);
        }
    }
}

// Might eventually want our own Format type, but for now this gives some convenience methods on
// vk::Format.
pub trait FormatExt {
    fn aspects(self) -> vk::ImageAspectFlags;
}

impl FormatExt for vk::Format {
    fn aspects(self) -> vk::ImageAspectFlags {
        match self {
            vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
                vk::ImageAspectFlags::DEPTH
            }

            vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,

            vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }

            _ => vk::ImageAspectFlags::COLOR,
        }
    }
}

pub fn subresource_range_full(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    // TODO: needs fixed for mips and arrays
    vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    pub format: vk::Format,
    pub extent: ImageExtent,
    pub tiling: ImageTiling,
    pub usage: vk::ImageUsageFlags,
}

impl From<ImageInfo> for vk::ImageCreateInfo<'static> {
    #[inline]
    fn from(value: ImageInfo) -> Self {
        vk::ImageCreateInfo::from(&value)
    }
}

impl From<&ImageInfo> for vk::ImageCreateInfo<'static> {
    fn from(value: &ImageInfo) -> Self {
        vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::empty())
            .image_type(value.extent.into())
            .format(value.format)
            .extent(value.extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(value.tiling.into())
            .usage(value.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&[])
            .initial_layout(vk::ImageLayout::UNDEFINED)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageExtent {
    D2(vk::Extent2D),
}

impl ImageExtent {
    pub fn as_2d(&self) -> Option<&vk::Extent2D> {
        match self {
            ImageExtent::D2(x2d) => Some(x2d),
        }
    }
}

impl From<(u32, u32)> for ImageExtent {
    #[inline]
    fn from((width, height): (u32, u32)) -> Self {
        ImageExtent::D2(vk::Extent2D { width, height })
    }
}

impl From<vk::Extent2D> for ImageExtent {
    #[inline]
    fn from(value: vk::Extent2D) -> Self {
        ImageExtent::D2(value)
    }
}

impl From<ImageExtent> for vk::ImageType {
    #[inline]
    fn from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D2(_) => vk::ImageType::TYPE_2D,
        }
    }
}

impl From<ImageExtent> for vk::Extent3D {
    #[inline]
    fn from(value: ImageExtent) -> Self {
        match value {
            ImageExtent::D2(x) => vk::Extent3D {
                width: x.width,
                height: x.height,
                depth: 1,
            },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageTiling {
    Optimal,
    Linear,
}

impl From<ImageTiling> for vk::ImageTiling {
    #[inline]
    fn from(value: ImageTiling) -> Self {
        match value {
            ImageTiling::Optimal => vk::ImageTiling::OPTIMAL,
            ImageTiling::Linear => vk::ImageTiling::LINEAR,
        }
    }
}

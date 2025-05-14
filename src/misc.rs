use std::time::Duration;

use ash::vk;

pub const IMAGE_SUBRESOURCE_RANGE_FULL_COLOR: vk::ImageSubresourceRange =
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    };

pub fn timeout_u64(duration: Option<Duration>) -> u64 {
    let Some(duration) = duration else {
        return u64::MAX;
    };

    duration
        .as_nanos()
        .try_into()
        .expect("timeout duration in nanoseconds should be less than u64::MAX")
}

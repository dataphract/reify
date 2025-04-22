use ash::vk;

pub struct Image {
    pub(crate) handle: vk::Image,
    pub(crate) mem: gpu_allocator::vulkan::Allocation,
}

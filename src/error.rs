#[derive(Debug)]
pub enum Error {
    /// Reify made a mistake.
    Bug,
    /// The container's capacity is exhausted.
    Capacity,
    /// The device was lost.
    DeviceLost,
    /// The operation timed out.
    TimedOut,
    Vulkan(ash::vk::Result),
}

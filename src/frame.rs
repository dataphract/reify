use std::time::Duration;

use ash::{prelude::*, vk};

use crate::{
    display::{DisplayInfo, SwapchainImage},
    misc::{timeout_u64, IMAGE_SUBRESOURCE_RANGE_FULL},
    Device,
};

/// Frame state that can be reused with different swapchain images.
pub struct FrameResources {
    /// Indicates whether the frame context is available for a new frame.
    ///
    /// The fence is unsignaled prior to queue submission, and signaled by the driver once all
    /// previous queue operations relying on this frame context have completed.
    context_available: vk::Fence,

    /// Indicates whether the associated swapchain image is available for rendering.
    ///
    /// The image may be acquired from the swapchain before it is safe to use as an attachment.
    /// This semaphore is unsignaled prior to queue submission and is signaled by the driver when
    /// the image can be safely used as an attachment.
    image_available: vk::Semaphore,

    /// Indicates whether all commands submitted in the current frame have completed.
    ///
    /// The semaphore is unsignaled prior to queue submission, and signaled by the driver once all
    /// graphics operations relying on this frame context have completed.
    // TODO(dp): split this into (render_complete, present_queue_ownership) when separate present
    // queue is supported
    all_commands_complete: vk::Semaphore,

    /// Dedicated command pool for the frame context.
    command_pool: vk::CommandPool,
    /// Command buffer for the frame context.
    commands: vk::CommandBuffer,
    // TODO(dp): cache intermediate targets
    //
    // /// Cache for intermediate targets.
    // image_cache: ImageCache,
}

impl FrameResources {
    pub fn create(device: &Device) -> FrameResources {
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let graphics_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.queue_family_index());

        unsafe {
            let context_available = device
                .create_fence(&fence_create_info)
                .expect("failed to create context_available fence");

            device
                .set_debug_utils_object_name(
                    &vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(context_available)
                        .object_name(c"context_available"),
                )
                .unwrap();

            let image_available = device
                .create_semaphore(&semaphore_create_info)
                .expect("failed to create image_available semaphore");
            device
                .set_debug_utils_object_name(
                    &vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(image_available)
                        .object_name(c"image_available"),
                )
                .unwrap();

            let render_complete = device
                .create_semaphore(&semaphore_create_info)
                .expect("failed to create render_complete semaphore");
            device
                .set_debug_utils_object_name(
                    &vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(render_complete)
                        .object_name(c"render_complete"),
                )
                .unwrap();

            let command_pool = device
                .create_command_pool(&graphics_pool_info)
                .expect("failed to create frame context graphics command pool");
            let cmdbuf_info = vk::CommandBufferAllocateInfo::default()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(1);
            let commands = device
                .allocate_command_buffers(&cmdbuf_info)
                .expect("failed to create frame context graphics command buffer")
                .pop()
                .unwrap();

            FrameResources {
                context_available,
                image_available,
                all_commands_complete: render_complete,
                command_pool,
                commands,
            }
        }
    }

    /// Blocks until the frame context is available or the timeout expires.
    #[tracing::instrument(skip_all)]
    pub fn acquire_context<'frame>(
        &'frame mut self,
        device: &Device,
        timeout: Option<Duration>,
    ) -> VkResult<AvailableFrameContext<'frame>> {
        // SAFETY: fences array is non-empty
        unsafe {
            device
                .wait_for_fences(&[self.context_available], true, timeout_u64(timeout))
                .map(|_| AvailableFrameContext { frame: self })
        }
    }
}

/// A `FrameContext` that is available for use, with no attached swapchain image.
pub struct AvailableFrameContext<'frame> {
    frame: &'frame mut FrameResources,
}

impl<'frame> AvailableFrameContext<'frame> {
    #[inline]
    pub fn image_available(&self) -> vk::Semaphore {
        self.frame.image_available
    }

    pub fn attach(
        self,
        device: &Device,
        display_info: &'frame DisplayInfo,
        swapchain: vk::SwapchainKHR,
        swapchain_image: &'frame mut SwapchainImage,
    ) -> FrameContext<'frame> {
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());

        unsafe {
            // SAFETY: context_available is signaled, so `self.commands` is not in the pending state.
            device
                .reset_command_pool(self.frame.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();

            device
                .begin_command_buffer(self.frame.commands, &begin_info)
                .unwrap();
        }

        // Record a barrier to:
        // 1. Order this frame's writes after the previous frame's.
        // 2. Transition the swapchain image back to COLOR_ATTACHMENT_OPTIMAL.
        let pre_render_barrier = vk::ImageMemoryBarrier2::default()
            // Order this frame's color output commands after the previous frame's.
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            // The whole image is being cleared, so it's not necessary to perform a QFOT here even
            // if the image was previously presented on another queue.
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_image.image)
            .subresource_range(IMAGE_SUBRESOURCE_RANGE_FULL);

        let image_memory_barriers = &[pre_render_barrier];

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);

        unsafe { device.cmd_pipeline_barrier2(self.frame.commands, &dependency_info) };

        FrameContext {
            device: device.clone(),
            display_info,
            resources: self.frame,
            attached: swapchain_image,
            swapchain,
        }
    }
}

pub struct FrameContext<'frame> {
    device: Device,
    display_info: &'frame DisplayInfo,
    resources: &'frame mut FrameResources,
    attached: &'frame mut SwapchainImage,
    swapchain: vk::SwapchainKHR,
}

impl<'frame> FrameContext<'frame> {
    #[tracing::instrument(skip_all)]
    pub fn submit_and_present(self, device: &Device) {
        // Record a barrier to:
        let post_render_barrier = vk::ImageMemoryBarrier2::default()
            // Order this frame's color output commands before the next frame's.
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            // Make this frame's writes to the color attachment available.
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::empty())
            // Transition to presentation layout.
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            // TODO(dp): update this when multiple queue families are supported
            //
            // Release ownership from graphics queue to present queue. If the queues are the
            // same, no ownership transfer occurs.
            .src_queue_family_index(device.queue_family_index())
            .dst_queue_family_index(device.queue_family_index())
            .image(self.attached.image)
            .subresource_range(IMAGE_SUBRESOURCE_RANGE_FULL);

        let image_memory_barriers = &[post_render_barrier];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);

        // Finish recording graphics commands.
        unsafe {
            device.cmd_pipeline_barrier2(self.resources.commands, &dep_info);

            device
                .end_command_buffer(self.resources.commands)
                .expect("failed to end recording graphics command buffer");
        }

        // Block rendering to the swapchain image until the driver is finished using it.
        let wait_semaphores = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.resources.image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

        let command_buffer_infos =
            &[vk::CommandBufferSubmitInfo::default().command_buffer(self.resources.commands)];

        let signal_semaphore_infos = &[vk::SemaphoreSubmitInfo::default()
            .semaphore(self.resources.all_commands_complete)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

        let submit_infos = &[vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_semaphores)
            .command_buffer_infos(command_buffer_infos)
            .signal_semaphore_infos(signal_semaphore_infos)];

        // Signal the context-available fence to allow this frame context to be reused once the
        // driver is done executing its commands.
        let signal_fence = self.resources.context_available;
        unsafe { device.reset_fences(&[signal_fence]).unwrap() };

        // Submit the command buffer.
        unsafe {
            device
                .queue()
                .submit2(submit_infos, Some(signal_fence))
                .unwrap()
        };

        let present_wait_semaphore = self.resources.all_commands_complete;

        let wait_semaphores = &[present_wait_semaphore];
        let swapchains = &[self.swapchain];
        let image_indices = &[self.attached.index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        unsafe {
            device
                .queue()
                .present(&present_info)
                .expect("failed to present swapchain image");
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn display_info(&self) -> &DisplayInfo {
        self.display_info
    }

    pub fn swapchain_image(&self) -> &SwapchainImage {
        &*self.attached
    }

    // TODO: shouldn't be a public API.
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.resources.commands
    }
}

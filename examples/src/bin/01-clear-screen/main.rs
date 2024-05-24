use ash::vk;

fn main() {
    examples::AppRunner::<ClearScreenApp>::new().run();
}

struct ClearScreenApp;

impl examples::App for ClearScreenApp {
    fn create_app(_device: &reify2::Device, _display_info: &reify2::DisplayInfo) -> ClearScreenApp {
        ClearScreenApp
    }

    fn render(&self, device: &reify2::Device, cx: &mut reify2::FrameContext) {
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(cx.swapchain_image().view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let color_attachments = &[color_attachment];

        let rendering_info = vk::RenderingInfo::default()
            .flags(vk::RenderingFlags::empty())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: cx.display_info().image_extent,
            })
            .layer_count(1)
            .view_mask(0)
            .color_attachments(color_attachments);

        unsafe {
            device.cmd_begin_rendering(cx.command_buffer(), &rendering_info);
            device.cmd_end_rendering(cx.command_buffer());
        }
    }
}

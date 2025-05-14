// This example demonstrates using a render pass to clear the screen.

fn main() {
    examples::AppRunner::<ClearScreenApp>::new().run();
}

struct ClearScreenApp {
    runtime: reify::Runtime,
}

impl examples::App for ClearScreenApp {
    fn create_app(device: &reify::Device, display_info: &reify::DisplayInfo) -> ClearScreenApp {
        let mut graph = reify::GraphEditor::new();

        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        graph
            .add_render_pass("clear_screen".into(), ClearScreenPass::default())
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify::Runtime::new(device.clone(), graph);

        ClearScreenApp { runtime }
    }

    fn runtime(&mut self) -> &mut reify::Runtime {
        &mut self.runtime
    }
}

struct ClearScreenPass {
    attachments: [reify::ColorAttachmentInfo; 1],
}

impl Default for ClearScreenPass {
    fn default() -> Self {
        Self {
            attachments: [reify::ColorAttachmentInfo {
                label: "out_color".into(),
                format: None,
                load_op: reify::LoadOp::Clear(reify::ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
                store_op: reify::StoreOp::Store,
            }],
        }
    }
}

impl reify::RenderPass for ClearScreenPass {
    fn color_attachments(&self) -> &[reify::ColorAttachmentInfo] {
        &self.attachments
    }
}

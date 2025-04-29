fn main() {
    examples::AppRunner::<ClearScreenApp>::new().run();
}

struct ClearScreenApp {
    runtime: reify2::Runtime,
}

struct ClearScreenPass {
    attachments: [reify2::ColorAttachmentInfo; 1],
}

impl Default for ClearScreenPass {
    fn default() -> Self {
        Self {
            attachments: [reify2::ColorAttachmentInfo {
                label: "out_color".into(),
                format: None,
                load_op: reify2::LoadOp::Clear(reify2::ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            }],
        }
    }
}

impl reify2::RenderPass for ClearScreenPass {
    fn color_attachments(&self) -> &[reify2::ColorAttachmentInfo] {
        &self.attachments
    }
}

impl examples::App for ClearScreenApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> ClearScreenApp {
        let mut graph = reify2::GraphBuilder::new();

        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify2::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        let _pass = graph
            .add_render_pass(ClearScreenPass::default())
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify2::Runtime::new(device.clone(), graph);

        ClearScreenApp { runtime }
    }

    fn render(&mut self, cx: &mut reify2::FrameContext) {
        self.runtime.execute(cx);
    }
}

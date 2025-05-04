use ash::vk;
use examples::{TrianglePipeline, TriangleRenderPass};
use reify2::BlitNode;

fn main() {
    examples::AppRunner::<BlitApp>::new().run();
}

struct BlitApp {
    runtime: reify2::Runtime,
}

impl examples::App for BlitApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> BlitApp {
        let triangle_pipeline = TrianglePipeline::new(
            display_info.surface_format.format,
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
        );

        let mut graph = reify2::GraphEditor::new();

        let triangle_out_color = graph.add_image(
            "triangle_out_color".into(),
            reify2::GraphImageInfo {
                format: vk::Format::B8G8R8A8_SRGB,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        graph
            .add_render_pass("triangle_pass".into(), TriangleRenderPass::default())
            .set_color_attachment("out_color".into(), triangle_out_color, None)
            .add_graphics_pipeline(device, triangle_pipeline)
            .build();

        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify2::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        graph.add_node(
            "blit".into(),
            BlitNode::new(triangle_out_color, swapchain_image, None),
        );

        let graph = graph.build(swapchain_image);
        let runtime = reify2::Runtime::new(device.clone(), graph);

        BlitApp { runtime }
    }

    fn runtime(&mut self) -> &mut reify2::Runtime {
        &mut self.runtime
    }
}

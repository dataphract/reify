// This example demonstrates drawing a triangle using a graphics pipeline.

use examples::{TrianglePipeline, TriangleRenderPass};

fn main() {
    examples::AppRunner::<TriangleApp>::new().run();
}

struct TriangleApp {
    runtime: reify2::Runtime,
}

impl examples::App for TriangleApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> TriangleApp {
        let triangle_pipeline = TrianglePipeline::new(
            display_info.surface_format.format,
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
        );

        let mut graph = reify2::GraphEditor::new();
        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify2::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        graph
            .add_render_pass("triangle_pass".into(), TriangleRenderPass::default())
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .add_graphics_pipeline(device, triangle_pipeline)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify2::Runtime::new(device.clone(), graph);

        TriangleApp { runtime }
    }

    fn runtime(&mut self) -> &mut reify2::Runtime {
        &mut self.runtime
    }
}

use std::ffi::CString;

use ash::vk;
use examples::GlslCompiler;
use naga::ShaderStage;

fn main() {
    examples::AppRunner::<TriangleApp>::new().run();
}

struct TriangleApp {
    runtime: reify2::Runtime,
}

struct TriangleRenderPass {
    color_attachments: [reify2::ColorAttachmentInfo; 1],
}

impl Default for TriangleRenderPass {
    fn default() -> Self {
        Self {
            color_attachments: [reify2::ColorAttachmentInfo {
                label: "out_color".into(),
                format: None,
                load_op: reify2::LoadOp::Clear(reify2::ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
            }],
        }
    }
}

impl reify2::RenderPass for TriangleRenderPass {
    fn color_attachments(&self) -> &[reify2::ColorAttachmentInfo] {
        &self.color_attachments
    }

    fn debug_label(&self) -> CString {
        c"triangle_pass".into()
    }
}

struct TrianglePipeline {
    color_format: vk::Format,
    vert_spv: Vec<u32>,
    frag_spv: Vec<u32>,
}

impl reify2::GraphicsPipeline for TrianglePipeline {
    fn vertex_info(&self) -> reify2::GraphicsPipelineVertexInfo {
        reify2::GraphicsPipelineVertexInfo {
            shader_spv: self.vert_spv.clone(),
            entry: c"main".into(),
        }
    }

    fn primitive_info(&self) -> reify2::GraphicsPipelinePrimitiveInfo {
        reify2::GraphicsPipelinePrimitiveInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }

    fn fragment_info(&self) -> reify2::GraphicsPipelineFragmentInfo {
        reify2::GraphicsPipelineFragmentInfo {
            shader_spv: self.frag_spv.clone(),
            entry: c"main".into(),
        }
    }

    fn attachment_info(&self) -> reify2::GraphicsPipelineAttachmentInfo {
        reify2::GraphicsPipelineAttachmentInfo {
            color: vec![self.color_format],
            depth: vk::Format::UNDEFINED,
            stencil: vk::Format::UNDEFINED,
        }
    }

    fn debug_label(&self) -> CString {
        c"triangle_pipeline".into()
    }

    fn execute(&self, pipe: &mut reify2::GraphicsPipelineInstance) {
        pipe.draw(3, 0);
    }
}

impl examples::App for TriangleApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> TriangleApp {
        let vert_src = r#"
#version 460 core

void main() {
    vec2 verts[3] = vec2[](
        vec2(-0.4, -0.5),
        vec2(0.4, -0.5),
        vec2(0.0, 0.5)
    );

    gl_Position = vec4(verts[gl_VertexIndex], 0.0, 1.0);
}
"#;

        let frag_src = r#"
#version 460 core

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.0, 0.2, 1.0, 1.0);
}
"#;

        let mut glslc = GlslCompiler::new();
        let vert_spv = glslc.compile(ShaderStage::Vertex, vert_src);
        let frag_spv = glslc.compile(ShaderStage::Fragment, frag_src);

        let triangle_pipeline = TrianglePipeline {
            color_format: display_info.surface_format.format,
            vert_spv,
            frag_spv,
        };

        let mut graph = reify2::GraphBuilder::new();
        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify2::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        graph
            .add_render_pass(TriangleRenderPass::default())
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .add_graphics_pipeline(device, triangle_pipeline)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify2::Runtime::new(device.clone(), graph);

        TriangleApp { runtime }
    }

    fn render(&mut self, cx: &mut reify2::FrameContext) {
        self.runtime.execute(cx);
    }
}

use std::ffi::CString;

use ash::vk;
use examples::GlslCompiler;
use naga::ShaderStage;
use reify::{ClearDepthStencilValue, StoreOp};

fn main() {
    examples::AppRunner::<DepthApp>::new().run();
}

// We'll draw three triangles from front to back to demonstrate that Z-buffering is working.
const TRIANGLES: [[[f32; 3]; 3]; 3] = [
    [[-0.8, -0.8, 0.2], [0.2, -0.8, 0.2], [-0.3, 0.2, 0.2]],
    [[-0.5, -0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.5, 0.4]],
    [[-0.2, -0.2, 0.6], [0.8, -0.2, 0.6], [0.3, 0.8, 0.6]],
];

const COLORS: [[f32; 3]; 3] = [[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.2, 0.2, 1.0]];

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

struct DepthApp {
    runtime: reify::Runtime,
}

impl examples::App for DepthApp {
    fn create_app(device: &reify::Device, display_info: &reify::DisplayInfo) -> DepthApp {
        let mut graph = reify::GraphEditor::new();

        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify::GraphImageInfo {
                format: display_info.surface_format.format,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        let depth_attachment = graph.add_image(
            "depth_attachment".into(),
            reify::GraphImageInfo {
                format: DEPTH_FORMAT,
                extent: *display_info.image_info.extent.as_2d().unwrap(),
            },
        );

        let color_format = display_info.surface_format.format;

        graph
            .add_render_pass("triangle_a".into(), DepthRenderPass::default())
            .add_graphics_pipeline(
                device,
                DepthPipeline::new(color_format, TRIANGLES[0], COLORS[0]),
            )
            .add_graphics_pipeline(
                device,
                DepthPipeline::new(color_format, TRIANGLES[1], COLORS[1]),
            )
            .add_graphics_pipeline(
                device,
                DepthPipeline::new(color_format, TRIANGLES[2], COLORS[2]),
            )
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .set_depth_stencil_attachment(depth_attachment, None)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify::Runtime::new(device.clone(), graph);

        DepthApp { runtime }
    }

    fn runtime(&mut self) -> &mut reify::Runtime {
        &mut self.runtime
    }
}

pub struct DepthRenderPass {
    color_attachments: [reify::ColorAttachmentInfo; 1],
    depth_attachment: reify::DepthStencilAttachmentInfo,
}

impl Default for DepthRenderPass {
    fn default() -> Self {
        Self {
            color_attachments: [reify::ColorAttachmentInfo {
                label: "out_color".into(),
                format: None,
                load_op: reify::LoadOp::Clear(reify::ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
                store_op: StoreOp::Store,
            }],
            depth_attachment: reify::DepthStencilAttachmentInfo {
                label: "out_depth".into(),
                format: Some(DEPTH_FORMAT),
                load_op: reify::LoadOp::Clear(ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                }),
                store_op: StoreOp::DontCare,
            },
        }
    }
}

impl reify::RenderPass for DepthRenderPass {
    fn color_attachments(&self) -> &[reify::ColorAttachmentInfo] {
        &self.color_attachments
    }

    fn depth_attachment(&self) -> Option<&reify::DepthStencilAttachmentInfo> {
        Some(&self.depth_attachment)
    }

    fn debug_label(&self) -> CString {
        c"triangle_pass".into()
    }
}

pub struct DepthPipeline {
    color_format: vk::Format,
    vert_spv: Vec<u32>,
    frag_spv: Vec<u32>,
}

impl DepthPipeline {
    pub fn new(
        color_format: vk::Format,
        vertices: [[f32; 3]; 3],
        color: [f32; 3],
    ) -> DepthPipeline {
        let vec4_glsl =
            |vert: [f32; 3]| format!("vec4({}, {}, {}, 1.0)", vert[0], vert[1], vert[2]);

        let [a, b, c] = vertices.map(vec4_glsl);
        let color = vec4_glsl(color);

        let vert_src = format!(
            r#"
#version 460 core

void main() {{
    vec4 verts[3] = vec4[](
        {a},
        {b},
        {c}
    );

    gl_Position = verts[gl_VertexIndex];
}}
"#
        );

        let frag_src = format!(
            r#"
#version 460 core

layout(location = 0) out vec4 out_color;

void main() {{
    out_color = {color};
}}
"#
        );

        // TODO: LazyLock
        let mut glslc = GlslCompiler::new();
        let vert_spv = glslc.compile(ShaderStage::Vertex, &vert_src);
        let frag_spv = glslc.compile(ShaderStage::Fragment, &frag_src);

        DepthPipeline {
            color_format,
            vert_spv,
            frag_spv,
        }
    }
}

impl reify::GraphicsPipeline for DepthPipeline {
    fn vertex_info(&self) -> reify::GraphicsPipelineVertexInfo {
        reify::GraphicsPipelineVertexInfo {
            shader_spv: self.vert_spv.clone(),
            entry: c"main".into(),
        }
    }

    fn primitive_info(&self) -> reify::GraphicsPipelinePrimitiveInfo {
        reify::GraphicsPipelinePrimitiveInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: false,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }

    fn fragment_info(&self) -> reify::GraphicsPipelineFragmentInfo {
        reify::GraphicsPipelineFragmentInfo {
            shader_spv: self.frag_spv.clone(),
            entry: c"main".into(),
        }
    }

    fn attachment_info(&self) -> reify::GraphicsPipelineAttachmentInfo {
        reify::GraphicsPipelineAttachmentInfo {
            color: vec![self.color_format],
            depth: DEPTH_FORMAT,
            stencil: vk::Format::UNDEFINED,
        }
    }

    fn depth_stencil_info(&self) -> reify::GraphicsPipelineDepthStencilInfo {
        reify::GraphicsPipelineDepthStencilInfo {
            depth_write_enable: true,
            compare_op: Some(vk::CompareOp::LESS),
        }
    }

    fn debug_label(&self) -> CString {
        c"triangle_pipeline".into()
    }

    fn execute(&self, pipe: &mut reify::GraphicsPipelineInstance) {
        pipe.draw(3, 0);
    }
}

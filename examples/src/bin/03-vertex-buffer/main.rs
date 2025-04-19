use std::{ffi::CString, mem};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use examples::GlslCompiler;

fn main() {
    examples::AppRunner::<VertexBufferApp>::new().run();
}

struct VertexBufferApp {
    runtime: reify2::Runtime,
}

#[derive(Copy, Clone, Debug, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    pos: [f32; 2],
}

impl examples::App for VertexBufferApp {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> VertexBufferApp {
        let mut upload_pool = reify2::UploadPool::create(
            device,
            &reify2::UploadPoolInfo {
                label: "upload_pool".into(),
                num_buffers: 1,
                buffer_size: 1024 * 1024,
            },
        );

        let verts = [
            Vertex { pos: [-0.4, -0.5] },
            Vertex { pos: [0.4, -0.5] },
            Vertex { pos: [0.0, 0.5] },
        ];

        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts[..]);

        let vertex_buffer = create_vertex_buffer(device, vert_bytes.len() as u64);

        unsafe {
            let upload_key = upload_pool
                .copy_bytes_to_buffer(
                    device,
                    vert_bytes,
                    mem::align_of::<Vertex>(),
                    vertex_buffer.handle,
                    device.transfer_queue_family_index(),
                )
                .unwrap();
        }

        todo!();

        let mut compiler = GlslCompiler::new();
        let vert_spv = compiler.compile(naga::ShaderStage::Vertex, VERT_SRC);
        let frag_spv = compiler.compile(naga::ShaderStage::Fragment, FRAG_SRC);

        let pipeline = VertexBufferPipeline {
            color_format: display_info.surface_format.format,
            vert_spv,
            frag_spv,
        };

        let mut graph = reify2::GraphBuilder::new();
        let swapchain_image = graph.add_image(
            "swapchain_image".into(),
            reify2::GraphImageInfo {
                // Infer format.
                format: None,
                // Infer extent.
                extent: None,
            },
        );

        graph
            .add_render_pass(VertexBufferRenderPass::default())
            .set_color_attachment("out_color".into(), swapchain_image, None)
            .add_graphics_pipeline(device, pipeline)
            .build();

        let graph = graph.build(swapchain_image);
        let runtime = reify2::Runtime::new(graph);

        VertexBufferApp { runtime }
    }

    fn render(&self, cx: &mut reify2::FrameContext) {
        self.runtime.execute(cx);
    }
}

struct VertexBuffer {
    handle: vk::Buffer,
    alloc: gpu_allocator::vulkan::Allocation,
}

fn create_vertex_buffer(device: &reify2::Device, size: u64) -> VertexBuffer {
    let buffer_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let handle = unsafe { device.create_buffer(&buffer_info).unwrap() };
    let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
    let alloc = device
        .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "vertex_buffer",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();

    unsafe {
        device
            .bind_buffer_memory(handle, alloc.memory(), 0)
            .unwrap()
    };

    VertexBuffer { handle, alloc }
}

struct VertexBufferRenderPass {
    color_attachments: [reify2::ColorAttachmentInfo; 1],
}

impl Default for VertexBufferRenderPass {
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

impl reify2::RenderPass for VertexBufferRenderPass {
    fn color_attachments(&self) -> &[reify2::ColorAttachmentInfo] {
        &self.color_attachments
    }

    fn debug_label(&self) -> CString {
        c"vertex_buffer_pass".into()
    }
}

struct VertexBufferPipeline {
    color_format: vk::Format,
    vert_spv: Vec<u32>,
    frag_spv: Vec<u32>,
}

impl reify2::GraphicsPipeline for VertexBufferPipeline {
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
        c"vertex_buffer_pipeline".into()
    }

    fn execute(&self, pipe: &mut reify2::GraphicsPipelineInstance<'_, '_>) {
        pipe.draw(3, 0);
    }
}

const VERT_SRC: &str = r#"
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

const FRAG_SRC: &str = r#"
#version 460 core

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.0, 0.2, 1.0, 1.0);
}
"#;

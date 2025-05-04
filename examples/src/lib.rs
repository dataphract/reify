use std::ffi::CString;

use ash::vk;
use naga::{back, front, valid, ShaderStage};
use tracing_subscriber::layer::SubscriberExt;
use winit::{
    event::WindowEvent, event_loop::ActiveEventLoop, platform::x11::EventLoopBuilderExtX11,
    window::Window,
};

pub trait App {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> Self;
    fn runtime(&mut self) -> &mut reify2::Runtime;
}

// Generic app runner.
//
// This handles the non-rendering application logic like windowing and event loop handling.
pub struct AppRunner<A> {
    device: reify2::Device,
    window: Option<Window>,
    display: Option<reify2::Display>,

    app: Option<A>,
}

impl<A: App> Default for AppRunner<A> {
    fn default() -> Self {
        AppRunner::new()
    }
}

impl<A: App> AppRunner<A> {
    pub fn new() -> AppRunner<A> {
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default().with(tracing_tracy::TracyLayer::default()),
        )
        .unwrap();

        pretty_env_logger::init();

        let phys_device = reify2::PhysicalDevice::new();
        let device = phys_device.create_device();

        AppRunner {
            device,
            window: None,
            display: None,
            app: None,
        }
    }

    pub fn run(&mut self) {
        // TODO handle wayland/x
        let event_loop = winit::event_loop::EventLoop::builder()
            .with_x11()
            .build()
            .unwrap();

        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        event_loop.run_app(self).unwrap();
    }

    pub fn create_window(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("reify2")
            .with_inner_size(winit::dpi::LogicalSize::new(1600, 900));

        let window = event_loop.create_window(attr).unwrap();
        let surface = reify2::create_surface(&window);

        let inner_size = window.inner_size();
        let extent = vk::Extent2D {
            width: inner_size.width,
            height: inner_size.height,
        };

        let display = unsafe { reify2::Display::create(&self.device, surface, extent) };

        self.app = Some(A::create_app(&self.device, display.info()));
        self.window = Some(window);
        self.display = Some(display);
    }

    pub fn redraw(&mut self) {
        let acquire = self
            .display
            .as_mut()
            .unwrap()
            .acquire_frame_context(&self.device);

        match acquire {
            Ok(mut cx) => {
                let rt = self.app.as_mut().unwrap().runtime();
                rt.execute(&mut cx);
                cx.submit_and_present(&self.device, rt.swapchain_image_layout());
            }

            Err(_) => self.recreate_display(),
        }
    }

    pub fn recreate_display(&mut self) {
        let inner_size = self.window.as_ref().unwrap().inner_size();
        let extent = vk::Extent2D {
            width: inner_size.width,
            height: inner_size.height,
        };

        unsafe {
            self.display
                .as_mut()
                .unwrap()
                .recreate(&self.device, extent)
        };
    }
}

impl<A: App> winit::application::ApplicationHandler for AppRunner<A> {
    #[inline]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.create_window(event_loop);
    }

    #[tracing::instrument(skip_all)]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                self.redraw();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

pub struct GlslCompiler {
    front: front::glsl::Frontend,
    validator: valid::Validator,
}

impl Default for GlslCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GlslCompiler {
    pub fn new() -> GlslCompiler {
        let glsl_front = front::glsl::Frontend::default();

        let validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        GlslCompiler {
            front: glsl_front,
            validator,
        }
    }

    pub fn compile(&mut self, stage: naga::ShaderStage, glsl: &str) -> Vec<u32> {
        let glsl_options = front::glsl::Options {
            stage,
            defines: Default::default(),
        };

        let parsed = self.front.parse(&glsl_options, glsl).unwrap();

        let info = self.validator.validate(&parsed).unwrap();

        let spv_options = back::spv::Options {
            lang_version: (1, 6),
            ..Default::default()
        };

        let pipe_opts = back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: "main".into(),
        };

        back::spv::write_vec(&parsed, &info, &spv_options, Some(&pipe_opts)).unwrap()
    }
}

pub struct TriangleRenderPass {
    color_attachments: [reify2::ColorAttachmentInfo; 1],
}

impl Default for TriangleRenderPass {
    fn default() -> Self {
        Self {
            color_attachments: [reify2::ColorAttachmentInfo {
                label: "out_color".into(),
                format: None,
                load_op: reify2::LoadOp::Clear(reify2::ClearColor::Float([0.0, 0.0, 0.0, 1.0])),
                store_op: reify2::StoreOp::Store,
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

pub struct TrianglePipeline {
    color_format: vk::Format,
    vert_spv: Vec<u32>,
    frag_spv: Vec<u32>,
}

impl TrianglePipeline {
    pub fn new(color_format: vk::Format, vertices: [[f32; 3]; 3]) -> TrianglePipeline {
        let vec3_glsl = |vert: [f32; 3]| format!("vec3({}, {}, {})", vert[0], vert[1], vert[2]);

        let [a, b, c] = vertices.map(vec3_glsl);

        let vert_src = format!(
            r#"
#version 460 core

void main() {{
    vec3 verts[3] = vec3[](
        {a},
        {b},
        {c}
    );

    gl_Position = vec4(verts[gl_VertexIndex], 1.0);
}}
"#
        );

        let frag_src = r#"
#version 460 core

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(0.0, 0.2, 1.0, 1.0);
}
"#;

        // TODO: LazyLock
        let mut glslc = GlslCompiler::new();
        let vert_spv = glslc.compile(ShaderStage::Vertex, &vert_src);
        let frag_spv = glslc.compile(ShaderStage::Fragment, frag_src);

        TrianglePipeline {
            color_format,
            vert_spv,
            frag_spv,
        }
    }
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

    fn depth_stencil_info(&self) -> reify2::GraphicsPipelineDepthStencilInfo {
        reify2::GraphicsPipelineDepthStencilInfo {
            depth_write_enable: false,
            compare_op: None,
        }
    }

    fn debug_label(&self) -> CString {
        c"triangle_pipeline".into()
    }

    fn execute(&self, pipe: &mut reify2::GraphicsPipelineInstance) {
        pipe.draw(3, 0);
    }
}

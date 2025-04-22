use ash::vk;
use naga::{back, front, valid};
use tracing_subscriber::layer::SubscriberExt;
use winit::{
    event::WindowEvent, event_loop::ActiveEventLoop, platform::x11::EventLoopBuilderExtX11,
    window::Window,
};

pub trait App {
    fn create_app(device: &reify2::Device, display_info: &reify2::DisplayInfo) -> Self;
    fn render(&mut self, cx: &mut reify2::FrameContext<'_>);
}

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
}

impl<A: App> winit::application::ApplicationHandler for AppRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::RedrawRequested => {
                let mut cx = self
                    .display
                    .as_mut()
                    .unwrap()
                    .acquire_frame_context(&self.device);

                self.app.as_mut().unwrap().render(&mut cx);

                cx.submit_and_present(&self.device);
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

use ash::vk;
use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent,
    event_loop::ActiveEventLoop, platform::x11::EventLoopBuilderExtX11, window::Window,
};

fn main() {
    pretty_env_logger::init();

    // TODO handle wayland/x
    let event_loop = winit::event_loop::EventLoop::builder()
        .with_x11()
        .build()
        .unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let phys_device = reify2::PhysicalDevice::new();
    let device = phys_device.create_device();

    let mut app = App {
        device,
        window: None,
        display: None,
    };

    event_loop.run_app(&mut app).unwrap();
}

struct App {
    device: reify2::Device,

    window: Option<Window>,
    display: Option<reify2::Display>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("reify2")
            .with_inner_size(LogicalSize::new(1600, 900));

        let window = event_loop.create_window(attr).unwrap();
        let surface = reify2::create_surface(&window);

        let inner_size = window.inner_size();
        let extent = vk::Extent2D {
            width: inner_size.width,
            height: inner_size.height,
        };

        let display = unsafe { reify2::Display::create(&self.device, surface, extent) };

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
                let cx = self
                    .display
                    .as_mut()
                    .unwrap()
                    .acquire_frame_context(&self.device);

                cx.submit_and_present(&self.device);
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

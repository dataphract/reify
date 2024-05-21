use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::WindowEvent,
    event_loop::ActiveEventLoop, platform::x11::EventLoopBuilderExtX11, window::Window,
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("reify2")
            .with_inner_size(LogicalSize::new(1600, 900));
        self.window = Some(event_loop.create_window(attr).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.window.as_ref().unwrap().request_redraw(),
            _ => (),
        }
    }
}

fn main() {
    pretty_env_logger::init();

    // TODO handle wayland/x
    let event_loop = winit::event_loop::EventLoop::builder()
        .with_x11()
        .build()
        .unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();

    let _instance = reify2::instance();

    event_loop.run_app(&mut app).unwrap();
}

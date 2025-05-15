//! Reify, a data-oriented rendering framework.

use std::sync::OnceLock;

use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

mod arena;

mod error;
pub use error::Error;

mod draw;

mod depgraph;

mod device;
pub use device::{Device, PhysicalDevice};

mod display;
pub use display::{Display, DisplayInfo};

mod frame;
pub use frame::FrameContext;

mod graph;
pub use graph::{builder::GraphEditor, Graph, GraphImageInfo, Runtime};

mod instance;
pub use instance::instance;

mod misc;

mod ops;
pub use ops::{
    blit::BlitNode,
    render_pass::{
        ClearColor, ClearDepthStencilValue, ColorAttachmentInfo, DepthStencilAttachmentInfo,
        GraphicsPipeline, GraphicsPipelineAttachmentInfo, GraphicsPipelineDepthStencilInfo,
        GraphicsPipelineFragmentInfo, GraphicsPipelineInstance, GraphicsPipelinePrimitiveInfo,
        GraphicsPipelineVertexInfo, LoadOp, OutputAttachmentInfo, RenderPass, RenderPassBuilder,
        StoreOp,
    },
};

mod pool;

mod resource;
pub use resource::{buffer, image, transient};

mod transfer;
pub use transfer::{UploadPool, UploadPoolInfo};

static ENTRY: OnceLock<ash::Entry> = OnceLock::new();

pub fn entry() -> &'static ash::Entry {
    ENTRY.get_or_init(|| unsafe { ash::Entry::load().expect("Vulkan loading failed") })
}

#[inline]
pub fn create_surface<T>(target: T) -> vk::SurfaceKHR
where
    T: HasDisplayHandle + HasWindowHandle,
{
    let display = target.display_handle().unwrap();
    let window = target.window_handle().unwrap();

    create_surface_impl(display.as_raw(), window.as_raw())
}

fn create_surface_impl(display: RawDisplayHandle, window: RawWindowHandle) -> vk::SurfaceKHR {
    let entry = entry();
    let instance = instance();

    let surface = match window {
        RawWindowHandle::Xlib(win) => {
            let RawDisplayHandle::Xlib(dpy) = display else {
                panic!("Xlib window combined with non-Xlib display");
            };

            let display = dpy.display.expect("Xlib display is null");
            let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                .flags(vk::XlibSurfaceCreateFlagsKHR::empty())
                .dpy(display.as_ptr())
                .window(win.window);

            // SAFETY: No external synchronization requirement.
            unsafe {
                // TODO: move to Instance
                khr::xlib_surface::Instance::new(entry, instance.instance())
                    .create_xlib_surface(&create_info, None)
                    .expect("failed to create Xlib window surface")
            }
        }

        RawWindowHandle::Xcb(win) => {
            let RawDisplayHandle::Xcb(dpy) = display else {
                panic!("XCB window combined with non-XCB display");
            };

            let conn = dpy.connection.expect("XCB connection is null");
            let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                .flags(vk::XcbSurfaceCreateFlagsKHR::empty())
                .connection(conn.as_ptr())
                .window(win.window.get());

            // SAFETY: No external synchronization requirement.
            unsafe {
                khr::xcb_surface::Instance::new(entry, instance.instance())
                    .create_xcb_surface(&create_info, None)
                    .expect("failed to create XCB window surface")
            }
        }

        RawWindowHandle::Wayland(win) => {
            let RawDisplayHandle::Wayland(dpy) = display else {
                panic!("Wayland window combined with non-Wayland display");
            };

            let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .flags(vk::WaylandSurfaceCreateFlagsKHR::empty())
                .display(dpy.display.as_ptr())
                .surface(win.surface.as_ptr());

            // SAFETY: No external synchronization requirement.
            unsafe {
                khr::wayland_surface::Instance::new(entry, instance.instance())
                    .create_wayland_surface(&create_info, None)
                    .expect("failed to create Wayland window surface")
            }
        }

        _ => panic!("unsupported window handle"),
    };

    surface
}

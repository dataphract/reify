use std::{
    any::Any,
    ffi::{CStr, CString},
};

use ash::vk;

use crate::{
    graph::{runtime::RunContext, GraphImage},
    image::ImageInfo,
    Device, FrameContext,
};

/// A trait for render graph nodes.
///
/// # Safety
///
/// Implementations are required to uphold Vulkan correctness requirements.
///
/// ...TODO
pub unsafe trait Node: Any {
    fn inputs(&self) -> NodeInputs {
        NodeInputs { images: &[] }
    }

    fn outputs(&self) -> NodeOutputs {
        NodeOutputs { images: &[] }
    }

    /// Returns the label used to annotate this node's debug spans.
    fn debug_label(&self) -> CString {
        c"[unlabeled node]".into()
    }

    unsafe fn execute(&self, cx: &mut NodeContext) {
        // Suppress unused param warning.
        let _ = cx;
    }
}

pub struct NodeInputs<'node> {
    pub images: &'node [InputImage],
}

pub(crate) struct OwnedNodeInputs {
    pub(crate) images: Vec<InputImage>,
}

impl OwnedNodeInputs {
    pub fn as_node_inputs(&self) -> NodeInputs {
        NodeInputs {
            images: &self.images,
        }
    }
}

pub struct InputImage {
    pub resource: GraphImage,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub usage: vk::ImageUsageFlags,
}

pub struct NodeOutputs<'node> {
    pub images: &'node [OutputImage],
}

#[derive(Default)]
pub(crate) struct OwnedNodeOutputs {
    pub(crate) images: Vec<OutputImage>,
}

impl OwnedNodeOutputs {
    pub fn as_node_outputs(&self) -> NodeOutputs {
        NodeOutputs {
            images: &self.images,
        }
    }
}

pub struct OutputImage {
    pub image: GraphImage,
    pub consumed: Option<GraphImage>,
    pub stage_mask: vk::PipelineStageFlags2,
    pub access_mask: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub usage: vk::ImageUsageFlags,
}

pub type BoxNode = Box<dyn Node>;

/// Node-local context during graph execution.
pub struct NodeContext<'run, 'frame> {
    run_cx: RunContext<'run>,
    frame: &'run mut FrameContext<'frame>,

    num_active_debug_spans: u32,
}

impl<'run, 'frame> Drop for NodeContext<'run, 'frame> {
    fn drop(&mut self) {
        while self.num_active_debug_spans > 0 {
            unsafe { self.exit_debug_span() };
        }
    }
}

impl<'run, 'frame> NodeContext<'run, 'frame> {
    pub(crate) fn new(
        run_cx: RunContext<'run>,
        frame: &'run mut FrameContext<'frame>,
    ) -> NodeContext<'run, 'frame> {
        NodeContext {
            run_cx,
            frame,
            num_active_debug_spans: 0,
        }
    }

    #[inline]
    pub fn device(&self) -> &Device {
        self.frame.device()
    }

    #[inline]
    pub fn image(&self, image: GraphImage) -> vk::Image {
        self.run_cx.image(image, self.frame)
    }

    #[inline]
    pub fn default_image_view(&self, image: GraphImage) -> vk::ImageView {
        self.run_cx.default_image_view(image, self.frame)
    }

    #[inline]
    pub fn image_info(&self, image: GraphImage) -> &ImageInfo {
        self.run_cx.image_info(image, self.frame)
    }

    #[inline]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.frame.command_buffer()
    }

    /// Enters a debug span in the command buffer.
    pub(crate) unsafe fn enter_debug_span(&mut self, label: &CStr, color: [f32; 4]) {
        unsafe { self.run_cx.enter_debug_span(self.frame, label, color) };
        self.num_active_debug_spans += 1;
    }

    /// Exits a debug span in the command buffer.
    pub(crate) unsafe fn exit_debug_span(&mut self) {
        if self.num_active_debug_spans == 0 {
            log::error!("called exit_debug_span() with no spans active.");
            return;
        }

        unsafe { self.run_cx.exit_debug_span(self.frame) };

        self.num_active_debug_spans -= 1;
    }

    pub(crate) unsafe fn exit_all_debug_spans(&mut self) {
        while self.num_active_debug_spans > 0 {
            log::warn!("terminating stray debug span");
            self.exit_debug_span();
        }
    }
}

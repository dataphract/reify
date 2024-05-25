use std::collections::HashMap;

use ash::vk;

use crate::{
    graph::{GraphImage, GraphKey, GraphNode, Node, NodeOutputs, OutputImage, OwnedNodeOutputs},
    FrameContext, GraphBuilder,
};

pub trait RenderPass {
    fn color_attachments(&self) -> &[ColorAttachmentInfo] {
        &[]
    }
}

pub struct RenderPassBuilder<'graph, R> {
    graph: &'graph mut GraphBuilder,

    pass: R,
    slots: RenderPassSlots,
}

impl<'graph, R> RenderPassBuilder<'graph, R>
where
    R: RenderPass + 'static,
{
    pub(crate) fn new(graph: &'graph mut GraphBuilder, pass: R) -> RenderPassBuilder<'graph, R> {
        RenderPassBuilder {
            graph,
            pass,
            slots: RenderPassSlots::default(),
        }
    }

    pub fn build(self) -> GraphKey {
        // TODO(dp): update capacity with other attachments
        let mut outputs = OwnedNodeOutputs {
            images: Vec::with_capacity(self.pass.color_attachments().len()),
        };

        for att in self.slots.color_attachments.values() {
            outputs.images.push(OutputImage {
                resource: att.produce,
                consumed: att.consume,
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            });
        }

        self.graph.add_node(GraphNode {
            node: Box::new(RenderPassNode {
                pass: self.pass,
                _slots: self.slots,
                outputs,
            }),
        })
    }

    pub fn set_color_attachment(
        mut self,
        label: String,
        image: GraphImage,
        consume: Option<GraphImage>,
    ) -> Self {
        if !self
            .pass
            .color_attachments()
            .iter()
            .enumerate()
            .any(|(_, slot)| slot.label == label)
        {
            panic!("No slot with the label `{label}`");
        };

        assert!(!self.slots.color_attachments.contains_key(&label));
        self.slots.color_attachments.insert(
            label,
            RenderPassOutputSlot {
                consume,
                produce: image,
            },
        );

        self
    }
}

#[derive(Default)]
struct RenderPassSlots {
    color_attachments: HashMap<String, RenderPassOutputSlot>,
}

struct RenderPassOutputSlot {
    pub(crate) consume: Option<GraphImage>,
    pub(crate) produce: GraphImage,
}

struct RenderPassNode<R> {
    pass: R,

    _slots: RenderPassSlots,
    outputs: OwnedNodeOutputs,
}

unsafe impl<R: RenderPass + 'static> Node for RenderPassNode<R> {
    fn outputs(&self) -> NodeOutputs {
        self.outputs.as_node_outputs()
    }

    unsafe fn execute(&self, cx: &mut FrameContext) {
        // TODO(dp): make method on ColorAttachmentInfo?
        let color_attachment_info = |att: &ColorAttachmentInfo| {
            let (load_op, clear_color) = match att.load_op {
                LoadOp::Load => (
                    vk::AttachmentLoadOp::LOAD,
                    vk::ClearColorValue { float32: [0.0; 4] },
                ),
                LoadOp::Clear(c) => (
                    vk::AttachmentLoadOp::CLEAR,
                    match c {
                        ClearColor::Float(f) => vk::ClearColorValue { float32: f },
                        ClearColor::SInt(s) => vk::ClearColorValue { int32: s },
                        ClearColor::UInt(u) => vk::ClearColorValue { uint32: u },
                    },
                ),
                LoadOp::DontCare => (
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::ClearColorValue { float32: [0.0; 4] },
                ),
            };

            vk::RenderingAttachmentInfo::default()
                .image_view(cx.swapchain_image().view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(load_op)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color })
        };

        let color_attachments = self
            .pass
            .color_attachments()
            .iter()
            .map(color_attachment_info)
            .collect::<Vec<_>>();

        let rendering_info = vk::RenderingInfo::default()
            .flags(vk::RenderingFlags::empty())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: cx.display_info().image_extent,
            })
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&color_attachments);

        // TODO
        // if let Some(depth) = depth_attachment {
        //     render_info = render_info.depth_attachment(&depth);
        // }
        // if let Some(stencil) = stencil_attachment {
        //     render_info = render_info.stencil_attachment(&stencil);
        // }

        unsafe {
            let device = cx.device();
            let cmdbuf = cx.command_buffer();
            device.cmd_begin_rendering(cmdbuf, &rendering_info);
        }

        // TODO run pipelines

        unsafe {
            cx.device().cmd_end_rendering(cx.command_buffer());
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum ClearColor {
    Float([f32; 4]),
    SInt([i32; 4]),
    UInt([u32; 4]),
}

#[derive(Copy, Clone, PartialEq)]
pub enum LoadOp<T> {
    Load,
    Clear(T),
    DontCare,
}

pub struct OutputAttachmentInfo<T> {
    /// The label of the output attachment.
    ///
    /// This label is used to uniquely identify the output attachment among all the node's outputs.
    pub label: String,

    /// The image format of the output attachment.
    ///
    /// If `None`, the format is inferred.
    pub format: Option<vk::Format>,

    /// The load operation used to load the attachment.
    pub load_op: LoadOp<T>,
}

pub type ColorAttachmentInfo = OutputAttachmentInfo<ClearColor>;

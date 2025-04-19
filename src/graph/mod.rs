use std::{any::Any, ffi::CString};

use ash::vk;

use crate::{
    arena::{self, Arena},
    depgraph::DepGraph,
    misc::IMAGE_SUBRESOURCE_RANGE_FULL,
    FrameContext,
};

pub(crate) mod builder;

pub struct Graph {
    swapchain_image: GraphImage,

    _image_info: Arena<GraphImageInfo>,

    graph: DepGraph<GraphKey, Dependency>,
    graph_order: Vec<arena::Key<GraphKey>>,

    nodes: Arena<GraphNode>,
}

impl Graph {
    pub fn execute(&self, cx: &mut FrameContext) {
        let device = cx.device().clone();
        let cmdbuf = cx.command_buffer();

        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        for &dep_key in self.graph_order.iter() {
            let node_key = self.graph.node(dep_key);
            let node = &self.nodes[*node_key];

            for dep in self.graph.outgoing_deps(dep_key) {
                for img in &dep.images {
                    image_barriers.push(self.image_barrier(cx, img));
                }
            }

            let deps = vk::DependencyInfo::default()
                .dependency_flags(vk::DependencyFlags::empty())
                .memory_barriers(&[])
                // TODO
                // .buffer_memory_barriers(&buffer_barriers)
                .image_memory_barriers(&image_barriers);

            unsafe {
                device.cmd_pipeline_barrier2(cmdbuf, &deps);

                cx.enter_debug_span(&node.node.debug_label(), [1.0, 0.6, 0.6, 1.0]);
                node.node.execute(cx);
                cx.exit_debug_span();
            }
        }
    }

    fn image_barrier(
        &self,
        cx: &FrameContext,
        img: &ImageDependency,
    ) -> vk::ImageMemoryBarrier2<'_> {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(img.src_stage_mask)
            .src_access_mask(img.src_access_mask)
            .dst_stage_mask(img.dst_stage_mask)
            .dst_access_mask(img.dst_access_mask)
            .old_layout(img.old_layout)
            .new_layout(img.new_layout)
            // TODO
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(if img.image == self.swapchain_image {
                cx.swapchain_image().image
            } else {
                todo!("load from context")
            })
            .subresource_range(if img.image == self.swapchain_image {
                IMAGE_SUBRESOURCE_RANGE_FULL
            } else {
                todo!("load from context")
            })
    }
}

/// A trait for render graph nodes.
///
/// # Safety
///
/// Implementations are required to uphold Vulkan correctness requirements.
///
/// ...TODO
pub unsafe trait Node: Any {
    fn outputs(&self) -> NodeOutputs {
        NodeOutputs { images: &[] }
    }

    /// Returns the label used to annotate this node's debug spans.
    fn debug_label(&self) -> CString {
        c"[unlabeled node]".into()
    }

    unsafe fn execute(&self, cx: &mut FrameContext) {
        // Suppress unused param warning.
        let _ = cx;
    }
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

pub struct GraphNode {
    pub(crate) node: Box<dyn Node>,
}

pub type GraphKey = arena::Key<GraphNode>;

pub(crate) struct OutputImage {
    pub(crate) resource: GraphImage,
    pub(crate) consumed: Option<GraphImage>,
    pub(crate) stage_mask: vk::PipelineStageFlags2,
    pub(crate) access_mask: vk::AccessFlags2,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) usage: vk::ImageUsageFlags,
}

pub type GraphImage = arena::Key<GraphImageInfo>;

#[derive(Debug)]
pub struct GraphImageInfo {
    pub format: Option<vk::Format>,
    pub extent: Option<vk::Extent2D>,
}

#[derive(Default)]
struct Dependency {
    images: Vec<ImageDependency>,
}

struct ImageDependency {
    image: GraphImage,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
}

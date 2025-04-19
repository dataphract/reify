use std::{any::Any, ffi::CString, sync::Arc};

use ash::vk;

use crate::{
    arena::{self, Arena},
    depgraph::DepGraph,
    FrameContext,
};

pub(crate) mod builder;

pub(crate) mod runtime;
pub use self::runtime::Runtime;

#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

struct GraphInner {
    swapchain_image: GraphImage,

    _image_info: Arena<GraphImageInfo>,

    graph: DepGraph<GraphKey, Dependency>,
    graph_order: Vec<arena::Key<GraphKey>>,

    nodes: Arena<GraphNode>,
}

impl Graph {
    fn image_barrier(
        &self,
        dep: &ImageDependency,
        img: vk::Image,
        range: vk::ImageSubresourceRange,
    ) -> vk::ImageMemoryBarrier2<'_> {
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(dep.src_stage_mask)
            .src_access_mask(dep.src_access_mask)
            .dst_stage_mask(dep.dst_stage_mask)
            .dst_access_mask(dep.dst_access_mask)
            .old_layout(dep.old_layout)
            .new_layout(dep.new_layout)
            // TODO
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img)
            .subresource_range(range)
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

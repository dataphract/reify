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
    fn node_dependencies(&self, node: arena::Key<GraphKey>) -> impl Iterator<Item = &Dependency> {
        self.inner.graph.outgoing_deps(node)
    }

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
    pub format: vk::Format,
    pub extent: vk::Extent2D,
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

impl ImageDependency {
    /// Creates a dependency to synchronize a write after a read.
    fn after_read(
        image: GraphImage,
        reader: &ImageAccess,
        writer: &ImageAccess,
    ) -> ImageDependency {
        ImageDependency {
            image,
            src_stage_mask: reader.stage_mask,
            dst_stage_mask: writer.stage_mask,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
            old_layout: reader.layout,
            new_layout: writer.layout,
        }
    }

    /// Creates a dependency to synchronize a write followed by a read or write.
    fn after_write(writer: &OutputImage, reader: &ImageAccess) -> ImageDependency {
        ImageDependency {
            image: writer.resource,
            src_stage_mask: writer.stage_mask,
            dst_stage_mask: reader.stage_mask,
            src_access_mask: writer.access_mask,
            dst_access_mask: reader.access_mask,
            old_layout: writer.layout,
            new_layout: reader.layout,
        }
    }
}

#[derive(Debug, Default)]
struct ImageAccesses {
    produced_by: Option<ImageAccess>,
    read_by: Vec<ImageAccess>,
    consumed_by: Option<ImageAccess>,
}

/// A record of a single image access.
#[derive(Debug)]
struct ImageAccess {
    /// The key of the node that performs the access.
    node_key: GraphKey,
    /// The stage in which the access occurs.
    stage_mask: vk::PipelineStageFlags2,
    /// The access type of the access.
    access_mask: vk::AccessFlags2,
    /// The required layout of the image when the access is performed.
    layout: vk::ImageLayout,
}

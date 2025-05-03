use std::sync::Arc;

use ash::vk;
use node::InputImage;

use crate::{
    arena::{self, Arena, ArenaMap},
    depgraph::DepGraph,
};

pub(crate) mod builder;

pub(crate) mod node;
use self::node::OutputImage;
pub use self::node::{BoxNode, Node};

pub(crate) mod runtime;
pub use self::runtime::Runtime;

#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

struct GraphInner {
    swapchain_image: GraphImage,

    // TODO(dp): consolidate?
    image_info: Arena<GraphImageInfo>,
    image_access: ArenaMap<GraphImage, ImageAccesses>,
    image_usage: ArenaMap<GraphImage, vk::ImageUsageFlags>,

    graph: DepGraph<GraphKey, NodeDependency>,
    graph_order: Vec<arena::Key<GraphKey>>,

    nodes: Arena<BoxNode>,
    node_labels: ArenaMap<GraphKey, String>,
}

impl Graph {
    #[inline]
    pub fn swapchain_image(&self) -> GraphImage {
        self.inner.swapchain_image
    }

    #[inline]
    pub fn num_images(&self) -> usize {
        self.inner.image_info.len()
    }

    fn node_dependencies(
        &self,
        node: arena::Key<GraphKey>,
    ) -> impl Iterator<Item = &NodeDependency> {
        self.inner.graph.outgoing_deps(node)
    }

    fn image_barrier(
        &self,
        dep: &ImageDependency,
        img: vk::Image,
        range: vk::ImageSubresourceRange,
    ) -> vk::ImageMemoryBarrier2<'_> {
        tracing_log::log::debug!("{:?} -> {:?}", dep.old_layout, dep.new_layout);
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

pub type GraphKey = arena::Key<BoxNode>;

pub type GraphImage = arena::Key<GraphImageInfo>;

#[derive(Debug)]
pub struct GraphImageInfo {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

/// A dependency between render graph nodes.
#[derive(Default)]
struct NodeDependency {
    images: Vec<ImageDependency>,
}

impl NodeDependency {
    fn add_image_read_after_write(&mut self, reader: &InputImage, writer: &ImageAccess) {
        self.images.push(ImageDependency {
            image: reader.resource,
            src_stage_mask: writer.stage_mask,
            dst_stage_mask: reader.stage_mask,
            src_access_mask: writer.access_mask,
            dst_access_mask: reader.access_mask,
            old_layout: writer.layout,
            new_layout: reader.layout,
        });
    }

    fn add_image_write_after_read(&mut self, writer: &OutputImage, reader: &ImageAccess) {
        self.images.push(ImageDependency {
            image: writer.image,
            src_stage_mask: reader.stage_mask,
            dst_stage_mask: writer.stage_mask,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
            old_layout: reader.layout,
            new_layout: writer.layout,
        });
    }

    fn add_image_write_after_write(&mut self, second: &OutputImage, first: &ImageAccess) {
        self.images.push(ImageDependency {
            image: second.image,
            src_stage_mask: first.stage_mask,
            dst_stage_mask: second.stage_mask,
            src_access_mask: first.access_mask,
            dst_access_mask: second.access_mask,
            old_layout: first.layout,
            new_layout: second.layout,
        });
    }
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

#[derive(Debug, Default)]
struct ImageAccesses {
    produced_by: Option<ImageAccess>,
    read_by: Vec<ImageAccess>,
    consumed_by: Option<ImageAccess>,
}

impl ImageAccesses {
    fn add_reader(&mut self, access: ImageAccess) {
        self.read_by.push(access);
    }

    fn produced_layout(&self) -> vk::ImageLayout {
        // TODO: this layout should be known for persistent images
        self.produced_by.map(|acc| acc.layout).unwrap()
    }

    fn read_layout(&self) -> vk::ImageLayout {
        let Some((first, rest)) = self.read_by.split_first() else {
            // No readers, so no layout transition needed.
            return self.produced_layout();
        };

        // TODO: check this once at compile time
        if rest.iter().any(|acc| acc.layout != first.layout) {
            panic!("image is read with multiple layouts");
        }

        first.layout
    }

    fn consumed_layout(&self) -> Option<vk::ImageLayout> {
        self.consumed_by.map(|acc| acc.layout)
    }
}

/// A record of a single image access.
#[derive(Copy, Clone, Debug)]
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

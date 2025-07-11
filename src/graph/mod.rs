use std::{
    ops::{BitOrAssign, Range},
    sync::Arc,
};

use ash::vk;

use crate::{
    arena::{self, Arena, ArenaMap},
    depgraph::DepGraph,
};

pub(crate) mod builder;

pub(crate) mod node;
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
    image_labels: ArenaMap<GraphImage, String>,

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
        log::debug!("{:?} -> {:?}", dep.old_layout, dep.new_layout);
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
pub type GraphBuffer = arena::Key<GraphBufferInfo>;
pub type GraphImage = arena::Key<GraphImageInfo>;

#[derive(Copy, Clone, Debug)]
pub struct GraphImageInfo {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

#[derive(Copy, Clone, Debug)]
pub struct GraphBufferInfo {
    pub size: u64,
}

/// A dependency between render graph nodes.
#[derive(Default)]
struct NodeDependency {
    // True iff this dependency connects operations in subsequent executions.
    is_backedge: bool,
    buffers: Vec<BufferDependency>,
    images: Vec<ImageDependency>,
}

impl NodeDependency {
    fn add_buffer_read_after_write(&mut self, reader: &InputBuffer, writer: &BufferAccess) {
        self.buffers.push(BufferDependency {
            buffer: reader.key,
            src_stage_mask: writer.stage_mask,
            dst_stage_mask: reader.access.stage_mask,
            src_access_mask: writer.access_mask,
            dst_access_mask: reader.access.access_mask,
        });
    }

    fn add_buffer_write_after_read(&mut self, writer: &OutputBuffer, reader: &BufferAccess) {
        self.buffers.push(BufferDependency {
            buffer: writer.key,
            src_stage_mask: reader.stage_mask,
            dst_stage_mask: writer.access.stage_mask,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
        });
    }

    fn add_buffer_write_after_write(&mut self, second: &OutputBuffer, first: &BufferAccess) {
        self.buffers.push(BufferDependency {
            buffer: second.key,
            src_stage_mask: first.stage_mask,
            dst_stage_mask: second.access.stage_mask,
            src_access_mask: first.access_mask,
            dst_access_mask: second.access.access_mask,
        });
    }

    fn add_image_read_after_write(&mut self, reader: &InputImage, writer: &ImageOpAccess) {
        self.images.push(ImageDependency {
            image: reader.key,
            src_stage_mask: writer.access.stage_mask,
            dst_stage_mask: reader.access.stage_mask,
            src_access_mask: writer.access.access_mask,
            dst_access_mask: reader.access.access_mask,
            old_layout: writer.access.layout,
            new_layout: reader.access.layout,
        });
    }

    fn add_image_write_after_read(&mut self, writer: &OutputImage, reader: &ImageOpAccess) {
        self.images.push(ImageDependency {
            image: writer.key,
            src_stage_mask: reader.access.stage_mask,
            dst_stage_mask: writer.access.stage_mask,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::empty(),
            old_layout: reader.access.layout,
            new_layout: writer.access.layout,
        });
    }

    fn add_image_write_after_write(&mut self, second: &OutputImage, first: &ImageOpAccess) {
        self.images.push(ImageDependency {
            image: second.key,
            src_stage_mask: first.access.stage_mask,
            dst_stage_mask: second.access.stage_mask,
            src_access_mask: first.access.access_mask,
            dst_access_mask: second.access.access_mask,
            old_layout: first.access.layout,
            new_layout: second.access.layout,
        });
    }
}

#[derive(Copy, Clone, Debug)]
struct BufferDependency {
    buffer: GraphBuffer,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
}

#[derive(Debug, Default)]
struct BufferAccesses {
    produced_by: Option<BufferAccess>,
    read_by: Vec<BufferAccess>,
    consumed_by: Option<BufferAccess>,
}

impl BufferAccesses {
    fn add_reader(&mut self, access: BufferAccess) {
        self.read_by.push(access);
    }
}

/// A record of a single buffer access.
#[derive(Copy, Clone, Debug, Default)]
struct BufferAccess {
    /// The stage in which the access occurs.
    stage_mask: vk::PipelineStageFlags2,
    /// The access type of the access.
    access_mask: vk::AccessFlags2,
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

#[derive(Default)]
struct ImageAccesses {
    produced_by: Option<ImageOpAccess>,
    read_by: Vec<ImageOpAccess>,
    consumed_by: Option<ImageOpAccess>,
}

impl ImageAccesses {
    fn add_reader(&mut self, access: ImageOpAccess) {
        self.read_by.push(access);
    }

    fn produced_layout(&self) -> vk::ImageLayout {
        // TODO: this layout should be known for persistent images
        self.produced_by.map(|acc| acc.access.layout).unwrap()
    }

    fn read_layout(&self) -> vk::ImageLayout {
        let Some((first, rest)) = self.read_by.split_first() else {
            // No readers, so no layout transition needed.
            return self.produced_layout();
        };

        // TODO: check this once at compile time
        if rest
            .iter()
            .any(|acc| acc.access.layout != first.access.layout)
        {
            panic!("image is read with multiple layouts");
        }

        first.access.layout
    }

    fn consumed_layout(&self) -> Option<vk::ImageLayout> {
        self.consumed_by.map(|acc| acc.access.layout)
    }
}

/// A record of a single image access.
#[derive(Copy, Clone, Debug, Default)]
pub struct ImageAccess {
    /// The stage in which the access occurs.
    pub stage_mask: vk::PipelineStageFlags2,
    /// The access type of the access.
    pub access_mask: vk::AccessFlags2,
    /// The required layout of the image when the access is performed.
    pub layout: vk::ImageLayout,
}

struct CompiledBufferAccess {
    stage_mask: vk::PipelineStageFlags2,
    access_mask: vk::AccessFlags2,
}

struct CompiledImageAccess {
    stage_mask: vk::PipelineStageFlags2,
    access_mask: vk::AccessFlags2,
    layout: vk::ImageLayout,
}

struct CompiledImageInfo {
    start: CompiledImageAccess,
    end: CompiledImageAccess,
}

struct BufferBarrier {
    src: CompiledBufferAccess,
    dst: CompiledBufferAccess,
}

struct ImageBarrier {
    src: CompiledImageAccess,
    dst: CompiledImageAccess,
}

struct Graph2 {
    instructions: Vec<Instr>,

    dependencies: Vec<Dependency>,
    buffer_barriers: Vec<BufferBarrier>,
    image_barriers: Vec<ImageBarrier>,
}

impl Graph2 {
    fn dependency(&self, idx: usize) -> &Dependency {
        &self.dependencies[idx]
    }
}

enum Instr {
    // TODO
    Op(GraphKey),
    /// Marks the beginning of a dependency.
    //
    // TODO(dp): technically the index here isn't necessary since it always goes 0,1,2,...
    Release,
    /// Marks the end of a dependency.
    Acquire(u32),
}

// TODO: size-optimize this
struct Dependency {
    buffer_range: Range<usize>,
    image_range: Range<usize>,
}

// Not intended to be part of the public API, just meant to cut down on code duplication. Buffers
// and images are treated largely the same way, with image layout being the largest distinction.
trait Resource {
    type Access: Copy + Clone + Default;

    type Usage: BitOrAssign + Copy + Default;
}

impl Resource for GraphBufferInfo {
    type Access = BufferAccess;

    type Usage = vk::BufferUsageFlags;
}

impl Resource for GraphImageInfo {
    type Access = ImageAccess;

    type Usage = vk::ImageUsageFlags;
}

struct Resources<R: Resource> {
    resources: Arena<R>,
    access: ArenaMap<arena::Key<R>, OpAccesses<R>>,
    usage: ArenaMap<arena::Key<R>, R::Usage>,
    labels: ArenaMap<arena::Key<R>, String>,
}

impl<R: Resource> Default for Resources<R> {
    fn default() -> Self {
        Resources {
            resources: Arena::default(),
            access: ArenaMap::default(),
            usage: ArenaMap::default(),
            labels: ArenaMap::default(),
        }
    }
}

impl<R: Resource> Resources<R> {
    fn len(&self) -> usize {
        self.resources.len()
    }

    fn add_resource(&mut self, label: String, info: R) -> arena::Key<R> {
        let key = self.resources.alloc(info);

        self.access.insert(key, OpAccesses::default());
        self.usage.insert(key, R::Usage::default());
        self.labels.insert(key, label);

        key
    }

    #[inline]
    fn add_usage(&mut self, key: arena::Key<R>, usage: R::Usage) {
        self.usage[key] |= usage;
    }
}

struct OpAccesses<R: Resource> {
    produced_by: Option<OpAccess<R>>,
    readers: Vec<OpAccess<R>>,
    consumed_by: Option<OpAccess<R>>,
}

impl<R: Resource> Default for OpAccesses<R> {
    fn default() -> Self {
        OpAccesses {
            produced_by: None,
            readers: Vec::new(),
            consumed_by: None,
        }
    }
}

impl<R: Resource> OpAccesses<R> {
    fn set_producer(&mut self, node_key: GraphKey, access: R::Access) {
        assert!(self.produced_by.is_none());

        self.produced_by = Some(OpAccess { node_key, access });
    }

    fn add_reader(&mut self, node_key: GraphKey, access: R::Access) {
        self.readers.push(OpAccess { node_key, access });
    }

    fn set_consumer(&mut self, node_key: GraphKey, access: R::Access) {
        assert!(self.consumed_by.is_none());

        self.consumed_by = Some(OpAccess { node_key, access });
    }
}

pub type ImageOpAccess = OpAccess<GraphImageInfo>;
pub type BufferOpAccess = OpAccess<GraphBufferInfo>;

#[derive(Copy, Clone)]
struct OpAccess<R: Resource> {
    node_key: GraphKey,
    access: R::Access,
}

pub type InputBuffer = Input<GraphBufferInfo>;
pub type InputImage = Input<GraphImageInfo>;

pub struct Input<R: Resource> {
    pub key: arena::Key<R>,
    pub access: R::Access,
    pub usage: R::Usage,
}

pub type OutputBuffer = Output<GraphBufferInfo>;
pub type OutputImage = Output<GraphImageInfo>;

pub struct Output<R: Resource> {
    pub key: arena::Key<R>,
    pub consumed: Option<arena::Key<R>>,
    pub access: R::Access,
    pub usage: R::Usage,
}

impl<R: Resource> Copy for Output<R> {}
impl<R: Resource> Clone for Output<R> {
    fn clone(&self) -> Self {
        Self {
            key: self.key,
            consumed: self.consumed,
            access: self.access,
            usage: self.usage,
        }
    }
}

use std::{
    any::Any,
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    ffi::CString,
    hash::Hash,
};

use ash::vk;

use crate::{
    arena::{self, Arena, ArenaMap},
    misc::IMAGE_SUBRESOURCE_RANGE_FULL,
    FrameContext, RenderPass, RenderPassBuilder,
};

pub struct Graph {
    swapchain_image: GraphImage,

    _image_info: Arena<GraphImageInfo>,

    graph: DependencyGraph,
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
            let node = &self.nodes[node_key];

            for dep in self.graph.outgoing_deps(dep_key) {
                for img in &dep.images {
                    image_barriers.push(
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
                            }),
                    );
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
}

// TODO(dp): maybe don't provide Default, since it guarantees implicit allocations?
#[derive(Default)]
pub struct GraphBuilder {
    images: Arena<GraphImageInfo>,
    image_access: ArenaMap<GraphImage, ImageAccesses>,
    image_usage: ArenaMap<GraphImage, vk::ImageUsageFlags>,
    image_labels: HashMap<String, GraphImage>,

    nodes: Arena<GraphNode>,
}

impl GraphBuilder {
    pub fn new() -> GraphBuilder {
        GraphBuilder::default()
    }

    pub fn build(self, final_image: GraphImage) -> Graph {
        let final_node = self
            .image_access
            .get(final_image)
            .unwrap()
            .produced_by
            .as_ref()
            .expect("can't set final image: not produced by any node")
            .node_key;

        // Build a dependency graph of nodes.
        //
        // Edges between nodes represent dependencies, which can be:
        //
        // - Memory dependencies, which ensure that a resource R produced by node A has all
        //   outstanding writes made available before being read by node B.
        // - Execution dependencies, which ensure that a resource R read by node A is not consumed
        //   by node B before A has finished its reads.

        // Insert memory dependencies.
        let mut deps = DependencyGraph::default();
        let mut next_depth = Vec::new();
        let mut cur_depth = HashSet::new();
        let mut consumers: HashMap<GraphKey, Vec<GraphImage>> = HashMap::new();
        let mut builder_to_compiled: ArenaMap<GraphKey, arena::Key<GraphKey>> = ArenaMap::default();

        next_depth.push(final_node);

        while !next_depth.is_empty() {
            assert!(cur_depth.is_empty());

            for builder_node in next_depth.drain(..) {
                if cur_depth.contains(&builder_node) {
                    // Another node at the previous depth already depends on this node.
                    continue;
                }

                builder_to_compiled.entry(builder_node).or_insert_with(|| {
                    cur_depth.insert(builder_node);
                    deps.add_node(builder_node)
                });
            }

            for builder_key in cur_depth.drain() {
                // Insert memory dependencies. Since entire depths are inserted into the graph at
                // once, this will not miss lateral edges (i.e., edges from one depth-N node to
                // another). Any node at this depth which hasn't been inserted into the dependency
                // graph is not a transitive dependency of the output node.
                let builder_node = &self.nodes[builder_key];

                for output in builder_node.node.outputs().images {
                    let produced_access = self.image_access.get(output.resource).unwrap();

                    if let Some(c) = output.consumed {
                        consumers.entry(builder_key).or_default().push(c);
                    }

                    for dependent in produced_access
                        .read_by
                        .iter()
                        .chain(produced_access.consumed_by.as_ref())
                    {
                        let Some(&src_key) = builder_to_compiled.get(dependent.node_key) else {
                            continue;
                        };

                        let dst_key = builder_to_compiled[builder_key];

                        deps.edge_mut_or_default(src_key, dst_key)
                            .images
                            .push(ImageDependency {
                                image: output.resource,
                                src_stage_mask: output.stage_mask,
                                dst_stage_mask: dependent.stage_mask,
                                src_access_mask: output.access_mask,
                                dst_access_mask: dependent.access_mask,
                                old_layout: output.layout,
                                new_layout: dependent.layout,
                            });
                    }
                }
            }
        }

        for (&consumer_key, consumeds) in consumers.iter() {
            let consumer_key = builder_to_compiled[consumer_key];

            for &consumed_key in consumeds {
                let access = &self.image_access[consumed_key];
                let consumer = access.consumed_by.as_ref().unwrap();
                for reader in &access.read_by {
                    let Some(&reader_key) = builder_to_compiled.get(reader.node_key) else {
                        continue;
                    };

                    deps.edge_mut_or_default(consumer_key, reader_key)
                        .images
                        .push(ImageDependency {
                            image: consumed_key,
                            src_stage_mask: reader.stage_mask,
                            dst_stage_mask: consumer.stage_mask,
                            src_access_mask: vk::AccessFlags2::empty(),
                            dst_access_mask: vk::AccessFlags2::empty(),
                            old_layout: reader.layout,
                            new_layout: consumer.layout,
                        });
                }
            }
        }

        let node_order = deps.toposort_reverse();

        Graph {
            swapchain_image: final_image,
            _image_info: self.images,
            graph: deps,
            graph_order: node_order,
            nodes: self.nodes,
        }
    }

    pub fn add_image(&mut self, label: String, info: GraphImageInfo) -> GraphImage {
        let key = self.images.alloc(info);
        self.image_access.insert(key, ImageAccesses::default());
        self.image_usage.insert(key, vk::ImageUsageFlags::default());

        match self.image_labels.entry(label) {
            // TODO(dp): overwrite? definitely don't panic
            Entry::Occupied(_) => panic!("duplicate label"),
            Entry::Vacant(v) => {
                v.insert(key);
            }
        }

        key
    }

    pub(crate) fn add_node(&mut self, node: GraphNode) -> GraphKey {
        self.nodes.alloc_with_key(|node_key| {
            // TODO(dp): node inputs

            let outputs = node.node.outputs();
            for output in outputs.images {
                // Update image usage.
                self.image_usage[output.resource] |= output.usage;

                // If this output consumes an image, mark the image as such.
                if let Some(consumed) = output.consumed {
                    let consumed_accesses = &mut self.image_access[consumed];

                    if consumed_accesses.consumed_by.is_some() {
                        panic!("resource ({consumed:?}) consumer is already set");
                    }

                    consumed_accesses.consumed_by = Some(ImageAccess {
                        node_key,
                        stage_mask: output.stage_mask,
                        access_mask: output.access_mask,
                        layout: output.layout,
                    });
                }

                let produced = output.resource;
                let produced_accesses = &mut self.image_access[produced];

                if produced_accesses.produced_by.is_some() {
                    panic!("resource ({produced:?}) producer is already set");
                }

                produced_accesses.produced_by = Some(ImageAccess {
                    node_key,
                    stage_mask: output.stage_mask,
                    access_mask: output.access_mask,
                    layout: output.layout,
                });

                println!("produced_accesses = {:?}", produced_accesses);
            }

            node
        })
    }

    pub fn add_render_pass<R>(&mut self, render_pass: R) -> RenderPassBuilder<'_, R>
    where
        R: RenderPass + 'static,
    {
        RenderPassBuilder::new(self, render_pass)
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

type EdgeKey = arena::Key<Edge>;

#[derive(Default)]
struct DependencyGraph {
    nodes: Arena<GraphKey>,
    adjacency: ArenaMap<arena::Key<GraphKey>, Vec<EdgeKey>>,
    inv_adjacency: ArenaMap<arena::Key<GraphKey>, Vec<EdgeKey>>,

    edges: Arena<Edge>,
    edge_map: HashMap<Edge, EdgeKey>,
    edge_weights: ArenaMap<EdgeKey, Dependency>,
}

impl DependencyGraph {
    pub fn add_node(&mut self, node: GraphKey) -> arena::Key<GraphKey> {
        let key = self.nodes.alloc(node);
        self.adjacency.insert(key, Vec::new());
        self.inv_adjacency.insert(key, Vec::new());
        key
    }

    pub fn add_edge(
        &mut self,
        src: arena::Key<GraphKey>,
        dst: arena::Key<GraphKey>,
        dependency: Dependency,
    ) -> EdgeKey {
        // Can't have circular dependencies.
        assert!(!self.edge_map.contains_key(&Edge { src: dst, dst: src }));

        debug_assert!(self.nodes.get(src).is_some());
        debug_assert!(self.nodes.get(dst).is_some());

        let edge = Edge { src, dst };
        let edge_key = self.edges.alloc(edge);
        self.edge_weights.insert(edge_key, dependency);

        self.adjacency[src].push(edge_key);
        self.inv_adjacency[dst].push(edge_key);
        self.edge_map.insert(edge, edge_key);

        edge_key
    }

    pub fn edge_mut_or_default(
        &mut self,
        src: arena::Key<GraphKey>,
        dst: arena::Key<GraphKey>,
    ) -> &mut Dependency {
        let edge = Edge { src, dst };

        if let Some(&edge_key) = self.edge_map.get(&edge) {
            return &mut self.edge_weights[edge_key];
        }

        let edge_key = self.add_edge(src, dst, Dependency::default());
        &mut self.edge_weights[edge_key]
    }

    pub fn node(&self, key: arena::Key<GraphKey>) -> GraphKey {
        self.nodes[key]
    }

    pub fn outgoing_deps(&self, key: arena::Key<GraphKey>) -> impl Iterator<Item = &Dependency> {
        self.adjacency[key]
            .iter()
            .map(|&edge_key| &self.edge_weights[edge_key])
    }

    pub fn toposort_reverse(&self) -> Vec<arena::Key<GraphKey>> {
        let mut sorted = VecDeque::with_capacity(self.nodes.len());

        // TODO(dp): use a bitvec
        let mut edges_seen: ArenaMap<EdgeKey, bool> = ArenaMap::default();

        // Initialize the queue with the list of nodes with no dependents.
        let mut queue = VecDeque::from_iter(
            self.inv_adjacency
                .iter()
                .filter_map(|(key, adj)| adj.is_empty().then_some(key)),
        );

        while let Some(node_key) = queue.pop_front() {
            sorted.push_front(node_key);

            for &adj in &self.adjacency[node_key] {
                // Mark each dependency.
                edges_seen[adj] = true;

                let dependency_key = self.edges[adj].dst;

                if self.inv_adjacency[dependency_key]
                    .iter()
                    .all(|&e| edges_seen[e])
                {
                    // Queue any dependent node whose dependents are all marked.
                    queue.push_back(dependency_key);
                }
            }
        }

        // If any edges have not been seen, the graph has at least one cycle.
        if edges_seen.iter().any(|(_, &b)| !b) {
            panic!("graph has cycles");
        }

        sorted.into()
    }
}

struct Edge {
    src: arena::Key<GraphKey>,
    dst: arena::Key<GraphKey>,
}

impl Clone for Edge {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Edge {}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src && self.dst == other.dst
    }
}

impl Eq for Edge {}

impl Hash for Edge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.src.hash(state);
        self.dst.hash(state);
    }
}

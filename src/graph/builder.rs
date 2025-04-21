use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::Arc,
};

use ash::vk;

use crate::{
    arena::{self, Arena, ArenaMap},
    depgraph::DepGraph,
    graph::{
        Graph, GraphImage, GraphImageInfo, GraphInner, GraphKey, GraphNode, ImageAccess,
        ImageAccesses, ImageDependency, NodeDependency,
    },
    RenderPass, RenderPassBuilder,
};

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
        let mut deps: DepGraph<GraphKey, NodeDependency> = DepGraph::default();
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
                            .push(ImageDependency::after_write(output, dependent));
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
                        .push(ImageDependency::after_read(consumed_key, reader, consumer));
                }
            }
        }

        let node_order = deps.toposort_reverse();

        Graph {
            inner: Arc::new(GraphInner {
                swapchain_image: final_image,
                _image_info: self.images,
                graph: deps,
                graph_order: node_order,
                nodes: self.nodes,
            }),
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

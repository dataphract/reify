use std::{collections::HashSet, mem, sync::Arc};

use crate::{
    arena::{self, Arena, ArenaMap},
    depgraph::DepGraph,
    graph::{
        BoxNode, Graph, GraphBuffer, GraphBufferInfo, GraphImage, GraphImageInfo, GraphInner,
        GraphKey, ImageOpAccess, Node, NodeDependency, Resources,
    },
    RenderPass, RenderPassBuilder,
};

// TODO(dp): maybe don't provide Default, since it guarantees implicit allocations?
#[derive(Default)]
pub struct GraphEditor {
    buffers: Resources<GraphBufferInfo>,
    images: Resources<GraphImageInfo>,

    nodes: Arena<BoxNode>,
    node_labels: ArenaMap<GraphKey, String>,
}

impl GraphEditor {
    pub fn new() -> GraphEditor {
        GraphEditor::default()
    }

    pub fn build(self, final_image: GraphImage) -> Graph {
        let final_node = self
            .images
            .access
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

        let graph = GraphBuilder::new(&self).build(final_node);
        let graph_order = graph.toposort_reverse();

        Graph {
            inner: Arc::new(GraphInner {
                swapchain_image: final_image,
                buffers: self.buffers,
                images: self.images,
                graph,
                graph_order,
                nodes: self.nodes,
                node_labels: self.node_labels,
            }),
        }
    }

    #[inline]
    pub fn add_buffer(&mut self, label: String, info: GraphBufferInfo) -> GraphBuffer {
        self.buffers.add_resource(label, info)
    }

    #[inline]
    pub fn add_image(&mut self, label: String, info: GraphImageInfo) -> GraphImage {
        self.images.add_resource(label, info)
    }

    #[inline]
    pub fn add_node<N: Node + 'static>(&mut self, label: String, node: N) -> GraphKey {
        // delegate to boxed impl to reduce monomorphization cost
        self.add_box_node(label, Box::new(node))
    }

    fn add_box_node(&mut self, label: String, node: BoxNode) -> GraphKey {
        let key = self.nodes.alloc_with_key(|node_key| {
            let inputs = node.inputs();
            let outputs = node.outputs();

            for input in inputs.images {
                self.images.add_usage(input.key, input.usage);
                self.images.access[input.key].add_reader(node_key, input.access);
            }

            for output in outputs.images {
                // Update image usage.
                self.images.add_usage(output.key, output.usage);

                // If this output consumes an image, mark the image as such.
                if let Some(consumed) = output.consumed {
                    let consumed_accesses = &mut self.images.access[consumed];

                    if consumed_accesses.consumed_by.is_some() {
                        panic!("resource ({consumed:?}) consumer is already set");
                    }

                    consumed_accesses.consumed_by = Some(ImageOpAccess {
                        node_key,
                        access: output.access,
                    });
                }

                let produced = output.key;
                let produced_accesses = &mut self.images.access[produced];

                if produced_accesses.produced_by.is_some() {
                    panic!("resource ({produced:?}) producer is already set");
                }

                produced_accesses.produced_by = Some(ImageOpAccess {
                    node_key,
                    access: output.access,
                });
            }

            node
        });

        self.node_labels.insert(key, label);

        key
    }

    pub fn add_render_pass<R>(&mut self, label: String, render_pass: R) -> RenderPassBuilder<'_, R>
    where
        R: RenderPass + 'static,
    {
        RenderPassBuilder::new(self, label, render_pass)
    }
}

struct GraphBuilder<'a> {
    editor: &'a GraphEditor,

    dep_graph: DepGraph<GraphKey, NodeDependency>,
    arena_to_depgraph: ArenaMap<GraphKey, arena::Key<GraphKey>>,
}

impl<'a> GraphBuilder<'a> {
    fn new(builder: &GraphEditor) -> GraphBuilder {
        GraphBuilder {
            editor: builder,
            dep_graph: DepGraph::default(),
            arena_to_depgraph: ArenaMap::default(),
        }
    }

    fn build(mut self, init: GraphKey) -> DepGraph<GraphKey, NodeDependency> {
        let mut cur_depth = HashSet::new();
        let mut next_depth = HashSet::new();

        let node_dep_key = self.dep_graph.add_node(init);
        self.arena_to_depgraph.insert(init, node_dep_key);
        cur_depth.insert(init);

        while !cur_depth.is_empty() {
            for node in cur_depth.drain() {
                self.visit(&mut next_depth, node);
            }

            mem::swap(&mut cur_depth, &mut next_depth);
        }

        self.dep_graph
    }

    fn visit(&mut self, next_depth: &mut HashSet<GraphKey>, node_key: GraphKey) {
        let node = &self.editor.nodes[node_key];

        // For each node input, add a dependency from this node to the producer of the input.
        for input in node.inputs().images {
            let accesses = self.editor.images.access.get(input.key).unwrap();
            let producer = &accesses.produced_by.expect("no producer for image");

            self.dependency_mut(next_depth, node_key, producer.node_key)
                .add_image_read_after_write(input, producer);
        }

        // For each node output, add dependencies if the output consumes an image.
        for output in node.outputs().images {
            let Some(consumed) = output.consumed else {
                continue;
            };

            let consumed_accesses = self.editor.images.access.get(consumed).unwrap();

            // If the consumed image has no readers, add a direct dependency on the image producer.
            if consumed_accesses.readers.is_empty() {
                let producer = consumed_accesses.produced_by.unwrap();
                self.dependency_mut(next_depth, node_key, producer.node_key)
                    .add_image_write_after_write(output, &producer);

                continue;
            }

            // Otherwise, add a dependency on each reader.
            for reader in &consumed_accesses.readers {
                self.dependency_mut(next_depth, node_key, reader.node_key)
                    .add_image_write_after_read(output, reader);
            }
        }
    }

    fn dependency_mut(
        &mut self,
        next_depth: &mut HashSet<GraphKey>,
        src: GraphKey,
        dst: GraphKey,
    ) -> &mut NodeDependency {
        let src_dep_key = *self.arena_to_depgraph.get(src).unwrap();
        let dst_dep_key = *self.arena_to_depgraph.entry(dst).or_insert_with(|| {
            next_depth.insert(dst);
            self.dep_graph.add_node(dst)
        });

        self.dep_graph.edge_mut_or_default(src_dep_key, dst_dep_key)
    }
}

use std::collections::HashMap;

use ash::vk;

use crate::{
    graph::{Graph, GraphImage},
    misc::IMAGE_SUBRESOURCE_RANGE_FULL,
    FrameContext,
};

pub struct Runtime {
    graph: Graph,

    // TODO: replace these hashmaps with arenas keyed on GraphImage

    // User-specified bindings to graph images.
    //
    // The default is ImageBinding::Transient, i.e., the runtime will allocate an image to use.
    bindings: HashMap<GraphImage, ImageBinding>,

    // Physical resource associated with each graph image.
    resolved: HashMap<GraphImage, ResolvedImage>,
}

impl Runtime {
    pub fn new(graph: Graph) -> Runtime {
        Runtime {
            graph,
            bindings: HashMap::new(),
            resolved: HashMap::new(),
        }
    }

    pub fn execute(&self, cx: &mut FrameContext) {
        let device = cx.device().clone();
        let cmdbuf = cx.command_buffer();

        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        for &dep_key in self.graph.inner.graph_order.iter() {
            let node_key = self.graph.inner.graph.node(dep_key);
            let node = &self.graph.inner.nodes[*node_key];

            // Synchronize all resource dependencies.
            for dep in self.graph.node_dependencies(dep_key) {
                for img_dep in &dep.images {
                    let img: vk::Image;
                    let range: vk::ImageSubresourceRange;

                    match self
                        .resolved
                        .get(&img_dep.image)
                        .expect("image not resolved")
                    {
                        ResolvedImage::Swapchain => {
                            img = cx.swapchain_image().image;
                            range = IMAGE_SUBRESOURCE_RANGE_FULL;
                        }
                    }

                    image_barriers.push(self.graph.image_barrier(img_dep, img, range));
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

#[derive(Default)]
enum ImageBinding {
    #[default]
    Transient,
    Swapchain,
}

enum ResolvedImage {
    Swapchain,
}

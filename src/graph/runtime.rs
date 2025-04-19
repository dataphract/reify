use ash::vk;

use crate::{graph::Graph, misc::IMAGE_SUBRESOURCE_RANGE_FULL, FrameContext};

pub struct Runtime {
    graph: Graph,
}

impl Runtime {
    pub fn new(graph: Graph) -> Runtime {
        Runtime { graph }
    }

    pub fn execute(&self, cx: &mut FrameContext) {
        let device = cx.device().clone();
        let cmdbuf = cx.command_buffer();

        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        for &dep_key in self.graph.inner.graph_order.iter() {
            let node_key = self.graph.inner.graph.node(dep_key);
            let node = &self.graph.inner.nodes[*node_key];

            for dep in self.graph.inner.graph.outgoing_deps(dep_key) {
                for img_dep in &dep.images {
                    let img: vk::Image;
                    let range: vk::ImageSubresourceRange;

                    if img_dep.image == self.graph.inner.swapchain_image {
                        img = cx.swapchain_image().image;
                        range = IMAGE_SUBRESOURCE_RANGE_FULL;
                    } else {
                        todo!("load from context")
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

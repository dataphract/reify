use std::ffi::CStr;

use ash::vk;

use crate::{
    arena::{self, ArenaMap},
    graph::{node::NodeContext, Graph, GraphImage, GraphImageInfo},
    image::{self, FormatExt, ImageInfo, ImageTiling},
    misc::IMAGE_SUBRESOURCE_RANGE_FULL_COLOR,
    transient::TransientResources,
    Device, FrameContext,
};

use super::GraphResource;

// TODO(dp): make configurable
const TRANSIENT_RESOURCE_INSTANCES: usize = 2;

/// Reusable graph runner.
pub struct Runtime {
    device: Device,
    graph: Graph,

    frame_counter: u64,
    num_active_debug_spans: u32,

    // User-specified bindings to graph images.
    //
    // The default is ImageBinding::Transient, i.e., the runtime will allocate an image to use.
    image_bindings: ImageBindings,

    // Swappable transient resource storage.
    transient: [TransientResources; TRANSIENT_RESOURCE_INSTANCES],
}

impl Runtime {
    // TODO(dp): maybe make this Device::create_runtime()?
    pub fn new(device: Device, graph: Graph) -> Runtime {
        let transient =
            [(); TRANSIENT_RESOURCE_INSTANCES].map(|_| TransientResources::new(&device));

        let mut image_bindings = ImageBindings::with_capacity(graph.num_images());

        image_bindings.set(graph.swapchain_image(), ImageBinding::Swapchain);

        Runtime {
            device,
            graph,
            frame_counter: 0,
            num_active_debug_spans: 0,
            image_bindings,
            transient,
        }
    }

    #[inline]
    pub fn transient_idx(&self) -> usize {
        self.frame_counter as usize % self.transient.len()
    }

    pub fn bind_image(&mut self, image: GraphImage, binding: ImageBinding) {
        self.image_bindings.set(image, binding);
    }

    // TODO: not super happy with this, but the swapchain is a special case
    pub fn swapchain_image_layout(&self) -> vk::ImageLayout {
        let key = self.graph.swapchain_image();
        self.graph
            .image_access(key)
            .produced_by
            .map(|acc| acc.access.layout)
            .unwrap()
    }

    #[tracing::instrument(skip_all)]
    fn resolve_resources(&mut self) {
        for (img_key, graph_img_info) in self.graph.inner.images.iter() {
            let img_info = ImageInfo {
                format: graph_img_info.format,
                extent: graph_img_info.extent.into(),
                tiling: ImageTiling::Optimal,
                usage: self.graph.inner.images.usage.get(img_key).cloned().unwrap(),
            };

            match self.image_bindings.get(img_key) {
                ImageBinding::Transient => {
                    let img = self.transient[self.transient_idx()]
                        .create(&self.device, img_key, &img_info)
                        .expect("failed to instantiate transient image");

                    let label = self.graph.inner.images.labels.get(img_key).unwrap();

                    unsafe {
                        self.device
                            .set_debug_utils_object_name_with(img.handle, format_args!("{}", label))
                            .unwrap()
                    };
                }

                ImageBinding::Swapchain => (),
            }
        }
    }

    #[tracing::instrument(name = "Runtime::execute", skip_all)]
    pub fn execute(&mut self, cx: &mut FrameContext) {
        self.resolve_resources();

        let device = cx.device().clone();
        let cmdbuf = cx.command_buffer();

        // TODO: cache this between executions
        let mut image_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        let mut run_cx = RunContext {
            device: &device,
            image_bindings: &self.image_bindings,
            transient: &self.transient[self.transient_idx()],
            num_active_debug_spans: 0,
        };

        for &dep_key in self.graph.inner.graph_order.iter() {
            image_barriers.clear();

            let node_key = self.graph.inner.graph.node(dep_key);
            let node = &self.graph.inner.nodes[*node_key];

            // For image outputs that don't consume an image, transition from UNDEFINED layout.
            for out in node.outputs().images {
                if out.consumed.is_some() {
                    continue;
                }

                let img;
                let range;
                match self.image_bindings.get(out.key) {
                    ImageBinding::Transient => {
                        let storage = self.transient[self.transient_idx()].get(out.key);
                        img = storage.handle;
                        range = image::subresource_range_full(storage.info.format.aspects());
                    }

                    ImageBinding::Swapchain => {
                        img = cx.swapchain_image().image;
                        range = IMAGE_SUBRESOURCE_RANGE_FULL_COLOR;
                    }
                }

                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(img)
                        // Don't reorder commands between frames.
                        .src_stage_mask(out.access.stage_mask)
                        .dst_stage_mask(out.access.stage_mask)
                        // Don't reorder accesses between frames.
                        .src_access_mask(out.access.access_mask)
                        .dst_access_mask(out.access.access_mask)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(out.access.layout)
                        .subresource_range(range),
                );
            }

            // Synchronize all resource dependencies.
            for dep in self.graph.node_dependencies(dep_key) {
                for img_dep in &dep.images {
                    let img: vk::Image;
                    let range: vk::ImageSubresourceRange;

                    match self.image_bindings.get(img_dep.image) {
                        ImageBinding::Transient => {
                            img = self.transient[self.transient_idx()]
                                .get(img_dep.image)
                                .handle;
                            range = IMAGE_SUBRESOURCE_RANGE_FULL_COLOR;
                        }
                        ImageBinding::Swapchain => {
                            img = cx.swapchain_image().image;
                            range = IMAGE_SUBRESOURCE_RANGE_FULL_COLOR;
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

                run_cx.enter_debug_span(cx, &node.debug_label(), [1.0, 0.6, 0.6, 1.0]);

                {
                    let mut node_cx = NodeContext::new(run_cx, cx);
                    node.execute(&mut node_cx);
                    node_cx.exit_all_debug_spans();
                }

                run_cx.exit_debug_span(cx);
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct RunContext<'run> {
    device: &'run Device,
    image_bindings: &'run ImageBindings,
    transient: &'run TransientResources,

    num_active_debug_spans: u32,
}

impl<'run> RunContext<'run> {
    pub fn image(&self, image: GraphImage, cx: &FrameContext) -> vk::Image {
        match self.image_bindings.get(image) {
            ImageBinding::Transient => self.transient.get(image).handle,
            ImageBinding::Swapchain => cx.swapchain_image().image,
        }
    }

    pub fn default_image_view(&self, image: GraphImage, cx: &FrameContext) -> vk::ImageView {
        match self.image_bindings.get(image) {
            ImageBinding::Transient => self.transient.get(image).default_view,
            ImageBinding::Swapchain => cx.swapchain_image().view(),
        }
    }

    pub fn image_info<'a>(&'a self, image: GraphImage, cx: &'a FrameContext) -> &'a ImageInfo {
        match self.image_bindings.get(image) {
            ImageBinding::Transient => &self.transient.get(image).info,
            ImageBinding::Swapchain => &cx.display_info().image_info,
        }
    }

    /// Enters a debug span in the command buffer.
    pub(crate) unsafe fn enter_debug_span(
        &mut self,
        cx: &mut FrameContext,
        label: &CStr,
        color: [f32; 4],
    ) {
        unsafe {
            self.device.cmd_begin_debug_utils_label(
                cx.command_buffer(),
                &vk::DebugUtilsLabelEXT::default()
                    .label_name(label)
                    .color(color),
            )
        };

        self.num_active_debug_spans += 1;
    }

    /// Exits a debug span in the command buffer.
    pub(crate) unsafe fn exit_debug_span(&mut self, cx: &mut FrameContext) {
        if self.num_active_debug_spans == 0 {
            log::error!("called exit_debug_span() with no spans active, ignoring.");
            return;
        }

        unsafe { self.device.cmd_end_debug_utils_label(cx.command_buffer()) };
    }
}

type ImageBindings = Bindings<GraphImageInfo>;

pub struct Bindings<T>
where
    T: GraphResource,
{
    bindings: ArenaMap<arena::Key<T>, T::Binding>,
}

impl<T> Bindings<T>
where
    T: GraphResource,
{
    fn with_capacity(cap: usize) -> Self {
        Bindings {
            bindings: ArenaMap::with_capacity(cap),
        }
    }

    fn get(&self, image: arena::Key<T>) -> T::Binding {
        self.bindings.get(image).copied().unwrap_or_default()
    }

    fn set(&mut self, image: arena::Key<T>, binding: T::Binding) {
        let _ = self.bindings.insert(image, binding);
    }
}

/// A logical binding for a graph image.
#[derive(Copy, Clone, Default, Debug)]
pub enum ImageBinding {
    /// Binds to a transient image managed by the runtime.
    ///
    /// The runtime will create a physical image on demand to satisfy the image parameters. The
    /// generated image resource persists between frames, and is recreated when the image parameters
    /// change.
    ///
    /// This value is the default, and is suitable for intermediate render targets whose contents do
    /// not need to be preserved between frames.
    #[default]
    Transient,
    /// Binds to the active swapchain image.
    Swapchain,
}

use std::collections::HashMap;

use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};

use crate::{
    graph::{Graph, GraphImage},
    image::Image,
    misc::{extent_2d_to_3d, IMAGE_SUBRESOURCE_RANGE_FULL},
    Device, FrameContext,
};

pub struct Runtime {
    device: Device,

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
    // TODO(dp): maybe make this Device::create_runtime()?
    pub fn new(device: Device, graph: Graph) -> Runtime {
        Runtime {
            device,
            graph,
            bindings: HashMap::new(),
            resolved: HashMap::new(),
        }
    }

    pub fn bind_image(&mut self, image: GraphImage, binding: ImageBinding) {
        self.bindings.insert(image, binding);
    }

    fn resolve_resources(&mut self) {
        for (img_key, _img_info) in self.graph.inner.image_info.iter() {
            match self.bindings.entry(img_key).or_default() {
                ImageBinding::Transient => {
                    self.resolved.entry(img_key).or_insert_with(|| {
                        let new_img = create_transient_image(&self.device, &self.graph, img_key);
                        ResolvedImage::Transient(new_img)
                    });
                }

                ImageBinding::Swapchain => {
                    self.resolved.insert(img_key, ResolvedImage::Swapchain);
                }
            }
        }
    }

    pub fn execute(&mut self, cx: &mut FrameContext) {
        self.resolve_resources();

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
                        ResolvedImage::Transient(t) => {
                            img = t.handle;
                            range = IMAGE_SUBRESOURCE_RANGE_FULL;
                        }

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

/// A logical binding for a graph image.
#[derive(Default)]
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

enum ResolvedImage {
    Transient(Image),
    Swapchain,
}

fn create_transient_image(device: &Device, graph: &Graph, img_key: GraphImage) -> Image {
    let img_info = graph.inner.image_info.get(img_key).unwrap();
    let accesses = graph.inner.image_access.get(img_key).unwrap();
    let usage = graph.inner.image_usage.get(img_key).unwrap();

    let families = &[device.graphics_queue().family().as_u32()];

    let create_info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::default())
        .image_type(vk::ImageType::TYPE_2D)
        .format(img_info.format)
        .extent(extent_2d_to_3d(img_info.extent))
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(*usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(families)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&create_info)
            .expect("image creation failed")
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let alloc_desc = AllocationCreateDesc {
        // TODO
        name: "",
        requirements,
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    };

    let mem = device.allocate(&alloc_desc).expect("allocation failed");

    unsafe {
        device
            .bind_image_memory(image, mem.memory(), 0)
            .expect("failed to bind image memory");
    }

    Image { handle: image, mem }
}

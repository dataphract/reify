use std::collections::{hash_map, HashMap};

use ash::vk;

use crate::{
    graph::GraphImage,
    image::{Image, ImageInfo},
    Device,
};

/// A set of transient resources used by a runtime.
pub struct TransientResources {
    /// Indicates whether all prior accesses to the contained resources have completed.
    available: vk::Semaphore,

    // TODO(dp): pool, not hashmap, and store key in binding
    images: HashMap<GraphImage, Image>,
}

impl TransientResources {
    pub fn new(device: &Device) -> Self {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let available = unsafe {
            device
                .create_semaphore(&semaphore_create_info)
                .expect("failed to create image_available semaphore")
        };

        unsafe {
            device
                .set_debug_utils_object_name(available, c"available")
                .unwrap()
        };

        TransientResources {
            available,
            images: HashMap::new(),
        }
    }

    /// Creates (or recreates) a transient image associated with `image`.
    pub fn create<'a>(
        &'a mut self,
        device: &Device,
        image: GraphImage,
        info: &ImageInfo,
    ) -> &'a Image {
        match self.images.entry(image) {
            hash_map::Entry::Occupied(mut o) => {
                if &o.get().info != info {
                    let old_image = o.insert(Image::create(device, info));
                    unsafe { old_image.destroy(device) };
                }

                &*o.into_mut()
            }

            hash_map::Entry::Vacant(v) => v.insert(Image::create(device, info)),
        }
    }

    pub fn get(&self, image: GraphImage) -> &Image {
        self.images.get(&image).expect("no associated image :(")
    }
}

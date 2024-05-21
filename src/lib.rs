use std::sync::OnceLock;

mod instance;
pub use instance::instance;

static ENTRY: OnceLock<ash::Entry> = OnceLock::new();

pub fn entry() -> &'static ash::Entry {
    ENTRY.get_or_init(|| unsafe { ash::Entry::load().expect("Vulkan loading failed") })
}

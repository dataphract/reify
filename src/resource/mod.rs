use std::iter;

use crate::{
    device::{OwnerId, Ownership},
    pool::{FreeList, RawPool},
    Device,
};

pub mod buffer;
pub mod image;
pub mod transient;

pub trait Resource {
    type CreateInfo;

    type OwnerState: Clone;

    type Handle: ash::vk::Handle + Copy;

    type Cold;

    fn from_key(key: ResourceKey) -> Self;
    fn key(&self) -> ResourceKey;
    fn create(device: &Device, info: Self::CreateInfo) -> (Self::Handle, Self::Cold);
    fn destroy(device: &Device, handle: Self::Handle, cold: Self::Cold);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceKey {
    generation: u32,
    index: u32,
}

impl ResourceKey {
    pub(crate) fn batch_order(self, other: Self) -> std::cmp::Ordering {
        self.index
            .cmp(&other.index)
            .then(self.generation.cmp(&other.generation))
    }
}

pub struct ResourcePool<T>
where
    T: Resource,
{
    free_list: FreeList,
    generation: Vec<u32>,
    ownership: RawPool<Ownership<T::OwnerState>>,
    handle: RawPool<T::Handle>,
    cold: RawPool<T::Cold>,
}

impl<T> ResourcePool<T>
where
    T: Resource,
{
    pub fn with_capacity(cap: usize) -> ResourcePool<T> {
        let free_list = FreeList::with_capacity(cap);
        let generations = iter::repeat(0).take(cap).collect();
        let ownership = RawPool::with_capacity(cap);
        let hot = RawPool::with_capacity(cap);
        let cold = RawPool::with_capacity(cap);

        ResourcePool {
            free_list,
            generation: generations,
            ownership,
            handle: hot,
            cold,
        }
    }

    #[inline]
    fn check_key(&self, key: ResourceKey) -> bool {
        key.generation == self.generation[key.index as usize]
    }

    // Attempts to set `owner` as the owner of the buffer associated with `key`.
    pub(crate) fn acquire(&mut self, resource: T, owner: OwnerId) -> Option<&T::OwnerState> {
        let key = resource.key();

        if !self.check_key(key) {
            return None;
        }

        let ownership = unsafe { self.ownership.get_mut(key.index as usize).assume_init_mut() };

        Some(ownership.acquire(owner))
    }

    // Attempts to release `owner`'s ownership of the buffer associated with `key`.
    pub(crate) fn release(&mut self, resource: T, owner: OwnerId, state: T::OwnerState) {
        let key = resource.key();

        if !self.check_key(key) {
            // TODO(dp): error
            panic!("stale key");
        }

        let ownership = unsafe { self.ownership.get_mut(key.index as usize).assume_init_mut() };

        ownership.release(owner, Some(state));
    }

    /// Returns the raw Vulkan handle to the resource.
    ///
    /// If the resource has been destroyed, returns `None`.
    ///
    /// # Safety
    ///
    /// This may only be called by the resource's current owner.
    //
    // TODO(dp): this shouldn't be exposed in the public API. Instead, it should be exposed as a
    // safe method on NodeContext, which calls this method.
    #[inline]
    pub unsafe fn handle(&self, resource: T) -> Option<T::Handle> {
        let key = resource.key();

        if !self.check_key(key) {
            return None;
        }

        // Safety: if a key with the right generation exists, the slot is initialized.
        unsafe { Some(self.handle.get(key.index as usize).assume_init()) }
    }

    /// Returns the resource's cold storage.
    ///
    /// If the resource has been destroyed, returns `None`.
    ///
    /// # Safety
    ///
    /// This may only be called by the resource's current owner.
    //
    // TODO(dp): this shouldn't be exposed in the public API. Instead, it should be exposed as a
    // safe method on NodeContext, which calls this method.
    pub unsafe fn cold(&self, key: ResourceKey) -> Option<&T::Cold> {
        if !self.check_key(key) {
            return None;
        }

        // Safety: if a key with the right generation exists, the slot is initialized.
        unsafe { Some(self.cold.get(key.index as usize).assume_init_ref()) }
    }

    /// Creates a new resource, returning its key.
    pub fn create(&mut self, device: &Device, info: T::CreateInfo) -> Option<T> {
        let index = self.free_list.pop()?;

        let (handle, cold) = T::create(device, info);

        self.handle.get_mut(index).write(handle);
        self.cold.get_mut(index).write(cold);

        Some(T::from_key(ResourceKey {
            generation: self.generation[index],
            index: index as u32,
        }))
    }

    pub fn destroy(&mut self, device: &Device, key: ResourceKey) {
        if !self.check_key(key) {
            panic!("already destroyed: {key:?}");
        }

        let index = key.index as usize;

        let handle = unsafe { self.handle.get_mut(index).assume_init_read() };
        let cold = unsafe { self.cold.get_mut(index).assume_init_read() };

        T::destroy(device, handle, cold);

        let ownership = unsafe { self.ownership.get(index).assume_init_ref() };

        if let Some(owner) = ownership.owner() {
            // TODO: get the owner label from the device, and return a proper error
            panic!("attempted to destroy resource still owned by {owner:?}");
        }

        self.generation[index] += 1;
        self.free_list.push(index);
    }
}

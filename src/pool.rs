use std::{
    mem::{self, MaybeUninit},
    num::NonZeroU32,
};

const MAX_LEN: usize = u32::MAX as usize - 1;

trait PoolKey {
    fn data(&self) -> PoolKeyData;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PoolKeyData {
    generation: u32,
    index: u32,
}

pub(crate) struct FreeList {
    head: Option<usize>,
    links: Vec<Link>,
}

impl FreeList {
    pub fn with_capacity(cap: usize) -> FreeList {
        // TODO: return error, don't assert
        assert!(cap > 0);
        assert!(cap <= MAX_LEN);

        let links = (0..cap - 1)
            .map(|i| Link::new(Some(i + 1)))
            .chain(std::iter::once_with(|| Link::new(None)))
            .collect();

        FreeList {
            head: Some(0),
            links,
        }
    }

    pub fn pop(&mut self) -> Option<usize> {
        let index = self.head?;

        self.head = self.links[index].take_next_free();

        Some(index)
    }

    pub fn push(&mut self, index: usize) {
        debug_assert!(self.links[index].next_free.is_none());

        let prev_head = self.head.replace(index);
        self.links[index].set_next_free(prev_head);
    }
}

struct Link {
    next_free: Option<NonZeroU32>,
}

impl Link {
    fn new(next_free: Option<usize>) -> Link {
        Link {
            next_free: next_free.and_then(|n| NonZeroU32::new(n as u32 + 1)),
        }
    }

    #[inline]
    fn take_next_free(&mut self) -> Option<usize> {
        self.next_free.take().map(|n| n.get() as usize - 1)
    }

    #[inline]
    fn set_next_free(&mut self, index: Option<usize>) {
        self.next_free = index.and_then(|n| NonZeroU32::new(n as u32 + 1));
    }
}

pub(crate) struct RawPool<V> {
    values: Vec<MaybeUninit<V>>,
}

impl<V> RawPool<V> {
    pub fn with_capacity(cap: usize) -> RawPool<V> {
        let mut values = Vec::with_capacity(cap);

        // Safety: Elements are MaybeUninit, so don't need to be initialized
        unsafe { values.set_len(cap) };

        RawPool { values }
    }

    #[inline]
    pub fn get(&self, index: usize) -> &MaybeUninit<V> {
        &self.values[index]
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut MaybeUninit<V> {
        &mut self.values[index]
    }
}

pub struct Gen<T> {
    value: T,
    gen: u32,
}

impl<T> Gen<T> {
    #[inline]
    pub fn new(value: T) -> Gen<T> {
        Gen { value, gen: 0 }
    }

    #[inline]
    pub fn gen(&self) -> u32 {
        self.gen
    }

    #[inline]
    pub fn get(&self) -> &T {
        &self.value
    }

    #[inline]
    pub fn set(&mut self, value: T) -> T {
        self.gen = self.gen.wrapping_add(1);
        mem::replace(&mut self.value, value)
    }
}

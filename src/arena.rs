// Based on the id-arena crate.

use std::{
    cmp, fmt,
    hash::Hash,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

pub(crate) struct Arena<T> {
    items: Vec<T>,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> Default for Arena<T> {
    #[inline]
    fn default() -> Self {
        Self {
            items: Vec::default(),
            _phantom: PhantomData,
        }
    }
}

impl<T> Index<Key<T>> for Arena<T> {
    type Output = T;

    #[inline]
    fn index(&self, key: Key<T>) -> &Self::Output {
        &self.items[key.index()]
    }
}

impl<T> IndexMut<Key<T>> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, key: Key<T>) -> &mut Self::Output {
        &mut self.items[key.index()]
    }
}

impl<T> Arena<T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn get(&self, key: Key<T>) -> Option<&T> {
        self.items.get(key.index())
    }

    #[inline]
    pub fn _get_mut(&mut self, key: Key<T>) -> Option<&mut T> {
        self.items.get_mut(key.index())
    }

    #[inline]
    pub fn alloc(&mut self, item: T) -> Key<T> {
        let key = self.next_key();

        self.items.push(item);

        key
    }

    #[inline]
    pub fn alloc_with_key<F>(&mut self, f: F) -> Key<T>
    where
        F: FnOnce(Key<T>) -> T,
    {
        let key = self.next_key();
        let value = f(key);
        self.alloc(value)
    }

    #[inline]
    fn next_key(&self) -> Key<T> {
        Key {
            index: self.items.len().try_into().expect("length overflowed u32"),
            _phantom: PhantomData,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Key<T>, &T)> {
        self.items.iter().enumerate().map(|(index, item)| {
            let key = Key {
                index: index as u32,
                _phantom: PhantomData,
            };

            (key, item)
        })
    }
}

pub struct Key<T> {
    index: u32,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> Key<T> {
    #[inline(always)]
    fn index(self) -> usize {
        self.index as usize
    }
}

impl<T> Clone for Key<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Key<T> {}

impl<T> fmt::Debug for Key<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Key").field("index", &self.index).finish()
    }
}

impl<T> PartialEq for Key<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Key<T> {}

impl<T> Hash for Key<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// Dense map type keyed by arena keys.
///
/// This type is designed for cases where all (or almost all) elements of an arena need some
/// associated data. Lookups are O(1), but a sparse `ArenaMap` wastes linear space in the number of
/// empty elements.
pub(crate) struct ArenaMap<K, V> {
    items: Vec<Option<V>>,
    _phantom: PhantomData<fn() -> K>,
}

impl<K, V> Default for ArenaMap<K, V> {
    fn default() -> Self {
        Self {
            items: Default::default(),
            _phantom: Default::default(),
        }
    }
}

impl<T, V> Index<Key<T>> for ArenaMap<Key<T>, V> {
    type Output = V;

    #[inline]
    fn index(&self, index: Key<T>) -> &Self::Output {
        self.items[index.index()]
            .as_ref()
            .unwrap_or_else(|| panic!("no entry with key {index:?}"))
    }
}

impl<T, V> IndexMut<Key<T>> for ArenaMap<Key<T>, V> {
    #[inline]
    fn index_mut(&mut self, index: Key<T>) -> &mut Self::Output {
        self.items[index.index()]
            .as_mut()
            .unwrap_or_else(|| panic!("no entry with key {index:?}"))
    }
}

impl<T, V> ArenaMap<Key<T>, V> {
    pub fn with_capacity(cap: usize) -> ArenaMap<Key<T>, V> {
        ArenaMap {
            items: Vec::with_capacity(cap),
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn grow(&mut self, key: Key<T>) {
        let min_len = key.index.checked_add(1).unwrap();
        let new_len = cmp::max(self.items.len(), min_len as usize);
        self.items.resize_with(new_len, || None);
    }

    #[inline]
    pub fn get(&self, key: Key<T>) -> Option<&V> {
        self.items.get(key.index())?.as_ref()
    }

    #[inline]
    pub fn _get_mut(&mut self, key: Key<T>) -> Option<&mut V> {
        self.items.get_mut(key.index())?.as_mut()
    }

    #[inline]
    pub fn insert(&mut self, key: Key<T>, value: V) -> Option<V> {
        self.grow(key);

        self.items[key.index()].replace(value)
    }

    #[inline]
    pub fn entry(&mut self, key: Key<T>) -> Entry<Key<T>, V> {
        self.grow(key);

        match self.items[key.index()] {
            Some(_) => Entry::Occupied(OccupiedEntry { map: self, key }),
            None => Entry::Vacant(VacantEntry { map: self, key }),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Key<T>, &V)> {
        self.items.iter().enumerate().filter_map(|(index, value)| {
            let value = value.as_ref()?;

            let key = Key {
                index: index as u32,
                _phantom: PhantomData,
            };

            Some((key, value))
        })
    }
}

pub(crate) enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, T, V> Entry<'a, Key<T>, V> {
    #[inline]
    pub fn or_insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(o) => o.map.items[o.key.index()].as_mut().unwrap(),
            Entry::Vacant(v) => v.insert_with(f),
        }
    }
}

pub(crate) struct OccupiedEntry<'a, K, V> {
    map: &'a mut ArenaMap<K, V>,
    key: K,
}

pub(crate) struct VacantEntry<'a, K, V> {
    map: &'a mut ArenaMap<K, V>,
    key: K,
}

impl<'a, T, V> VacantEntry<'a, Key<T>, V> {
    #[inline]
    pub fn insert_with<F>(self, f: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        self.map.items[self.key.index()].insert(f())
    }
}

//! Generic dependency graph.

use std::collections::{HashMap, VecDeque};

use crate::arena::{self, Arena, ArenaMap};

pub(crate) type EdgeKey<V> = arena::Key<Edge<V>>;

pub(crate) struct DepGraph<V, E> {
    nodes: Arena<V>,
    adjacency: ArenaMap<arena::Key<V>, Vec<EdgeKey<V>>>,
    inv_adjacency: ArenaMap<arena::Key<V>, Vec<EdgeKey<V>>>,

    edges: Arena<Edge<V>>,
    edge_map: HashMap<Edge<V>, EdgeKey<V>>,
    edge_weights: ArenaMap<EdgeKey<V>, E>,
}

impl<V, E> Default for DepGraph<V, E> {
    fn default() -> Self {
        DepGraph {
            nodes: Arena::default(),
            adjacency: ArenaMap::default(),
            inv_adjacency: ArenaMap::default(),

            edges: Arena::default(),
            edge_map: HashMap::default(),
            edge_weights: ArenaMap::default(),
        }
    }
}

impl<V, E> DepGraph<V, E> {
    pub fn add_node(&mut self, node: V) -> arena::Key<V> {
        let key = self.nodes.alloc(node);
        self.adjacency.insert(key, Vec::new());
        self.inv_adjacency.insert(key, Vec::new());
        key
    }

    pub fn add_edge(
        &mut self,
        src: arena::Key<V>,
        dst: arena::Key<V>,
        dependency: E,
    ) -> EdgeKey<V> {
        // Can't have circular dependencies.
        assert!(!self.edge_map.contains_key(&Edge { src: dst, dst: src }));

        debug_assert!(self.nodes.get(src).is_some());
        debug_assert!(self.nodes.get(dst).is_some());

        let edge = Edge { src, dst };
        let edge_key = self.edges.alloc(edge);
        self.edge_weights.insert(edge_key, dependency);

        self.adjacency[src].push(edge_key);
        self.inv_adjacency[dst].push(edge_key);
        self.edge_map.insert(edge, edge_key);

        edge_key
    }

    pub fn node(&self, key: arena::Key<V>) -> &V {
        &self.nodes[key]
    }

    pub fn outgoing_deps(&self, key: arena::Key<V>) -> impl Iterator<Item = &E> {
        self.adjacency[key]
            .iter()
            .map(|&edge_key| &self.edge_weights[edge_key])
    }

    // Performs a topological sort of the graph, returning the reversed result.
    pub fn toposort_reverse(&self) -> Vec<arena::Key<V>> {
        log::debug!("toposorting {} nodes", self.nodes.len());

        let mut sorted = VecDeque::with_capacity(self.nodes.len());

        // TODO(dp): use a bitvec
        let mut edges_seen: ArenaMap<EdgeKey<V>, bool> = ArenaMap::with_capacity(self.nodes.len());

        // Initialize the queue with the list of nodes with no dependents.
        let mut queue = VecDeque::from_iter(
            self.inv_adjacency
                .iter()
                .filter_map(|(key, adj)| adj.is_empty().then_some(key)),
        );

        while let Some(node_key) = queue.pop_front() {
            sorted.push_front(node_key);

            for &adj in &self.adjacency[node_key] {
                // Mark each dependency.
                edges_seen.insert(adj, true);

                let dependency_key = self.edges[adj].dst;

                if self.inv_adjacency[dependency_key]
                    .iter()
                    .all(|&e| edges_seen[e])
                {
                    // Queue any dependent node whose dependents are all marked.
                    queue.push_back(dependency_key);
                }
            }
        }

        // If any edges have not been seen, the graph has at least one cycle.
        if edges_seen.iter().any(|(_, &b)| !b) {
            panic!("graph has cycles");
        }

        sorted.into()
    }
}

impl<V, E> DepGraph<V, E>
where
    E: Default,
{
    /// Returns a mutable reference to the edge between `src` and `dst`, inserting a default value
    /// if no edge exists.
    pub fn edge_mut_or_default(&mut self, src: arena::Key<V>, dst: arena::Key<V>) -> &mut E {
        let edge = Edge { src, dst };

        if let Some(&edge_key) = self.edge_map.get(&edge) {
            return &mut self.edge_weights[edge_key];
        }

        let edge_key = self.add_edge(src, dst, E::default());
        &mut self.edge_weights[edge_key]
    }
}

pub(crate) struct Edge<V> {
    src: arena::Key<V>,
    dst: arena::Key<V>,
}

impl<V> Clone for Edge<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for Edge<V> {}

impl<V> PartialEq for Edge<V> {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src && self.dst == other.dst
    }
}

impl<V> Eq for Edge<V> {}

impl<V> std::hash::Hash for Edge<V> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.src.hash(state);
        self.dst.hash(state);
    }
}

use crate::lib::decision_process::SimpleMovingAverage;
use crate::lib::mcgs::common::Internal;
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use dashmap::DashMap;
use parking_lot::lock_api::RawMutex;
use parking_lot::RawMutex as Mutex;
use std::cell::UnsafeCell;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;

// Graph as of now suffers from stability issues
// the value of nodes has a lot of random flail

pub struct Node<O, I> {
    lock: Mutex,
    internal: UnsafeCell<Internal<O>>,
    edges: Vec<Edge<O, I>>,
}

pub struct Edge<O, I> {
    data: I,
    internal: UnsafeCell<Internal<O>>,
    node: UnsafeCell<Option<Arc<Node<O, I>>>>,
}

impl<O: Clone, I> Edge<O, I> {
    fn new(i: I, outcome: O) -> Self {
        Edge {
            data: i,
            internal: UnsafeCell::new(Internal::new(outcome.clone())),
            node: UnsafeCell::new(None),
        }
    }
}

unsafe impl<O, I> Sync for Node<O, I> {}

impl<I, O: Clone + SimpleMovingAverage> OutcomeStore<O> for Node<O, I> {
    fn expected_outcome(&self) -> O {
        unsafe { &*self.internal.get() }.expected_sample.clone()
    }

    fn is_solved(&self) -> bool {
        unsafe { &*self.internal.get() }.is_solved()
    }

    fn add_sample(&self, outcome: &O, weight: u32) {
        unsafe { &mut *self.internal.get() }.add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.sample_count
    }

    fn mark_solved(&self, outcome: &O) {
        unsafe { &mut *self.internal.get() }.fix(outcome)
    }
}

impl<I, O: Clone + SimpleMovingAverage> OutcomeStore<O> for Edge<O, I> {
    fn expected_outcome(&self) -> O {
        unsafe { &*self.internal.get() }.expected_sample.clone()
    }

    fn is_solved(&self) -> bool {
        unsafe { &*self.internal.get() }.is_solved()
    }

    fn add_sample(&self, outcome: &O, weight: u32) {
        unsafe { &mut *self.internal.get() }.add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.sample_count
    }

    fn mark_solved(&self, outcome: &O) {
        unsafe { &mut *self.internal.get() }.fix(outcome)
    }
}

impl<O, I> SelectCountStore for Node<O, I> {
    fn selection_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.selection_count
    }

    fn increment_selection_count(&self) {
        unsafe { &mut *self.internal.get() }.selection_count += 1
    }
}

impl<O, I> SelectCountStore for Edge<O, I> {
    fn selection_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.selection_count
    }

    fn increment_selection_count(&self) {
        unsafe { &mut *self.internal.get() }.selection_count += 1
    }
}

impl<O, I> ConcurrentAccess for Node<O, I> {
    fn lock(&self) {
        self.lock.lock()
    }

    fn unlock(&self) {
        unsafe { self.lock.unlock() }
    }
}

impl<O, I> ConcurrentAccess for Edge<O, I> {
    // Locks not needed as we use the locks from the node
    fn lock(&self) {}
    fn unlock(&self) {}
}

pub struct SafeGraph<H: Eq + Hash, D, O> {
    default_outcome: O,
    nodes: DashMap<H, Arc<Node<O, D>>>,
}

impl<H: Eq + Hash, D, O> SafeGraph<H, D, O> {
    pub(crate) fn new(outcome: O) -> Self {
        // TODO: set capacity
        SafeGraph {
            default_outcome: outcome,
            nodes: DashMap::new(),
        }
    }
}

pub trait Hsh<S> {
    type K;
    fn key(&self, s: &S) -> Self::K;
}

pub struct NoHash;
impl<S> Hsh<S> for NoHash {
    type K = ();
    fn key(&self, _: &S) -> () {}
}

impl<H: Eq + Hash, O: Clone, D> SearchGraph<D, H> for SafeGraph<H, D, O> {
    type Node = Node<O, D>;
    type Edge = Edge<O, D>;
    type NodeRef = Arc<Self::Node>;

    fn create_node<L: Iterator<Item = D>>(&self, s: H, l: L) -> Self::NodeRef {
        let opt = self.nodes.get(&s);
        if opt.is_none() {
            let edges = l
                .map(|e| Edge::new(e, self.default_outcome.clone()))
                .collect();
            let n = Arc::new(Node::new(self.default_outcome.clone(), edges));
            self.nodes.insert(s, n.clone());
            n
        } else {
            opt.unwrap().clone()
        }
    }

    fn add_child<'a, L: Iterator<Item = D>>(
        &self,
        s: H,
        e: &'a Self::Edge,
        l: L,
    ) -> &'a Self::Node {
        let n = self.create_node(s, l);
        unsafe {
            let eo = &mut *e.node.get();
            if eo.is_none() {
                eo.replace(n);
            }
            eo.as_ref().unwrap()
        }
    }

    fn clear(&self, _: Self::NodeRef) {
        self.nodes.clear()
    }

    fn is_leaf(&self, n: &Self::Node) -> bool {
        n.edges.is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        n.edges.len() as u32
    }

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge {
        n.edges.get(ix as usize).unwrap()
    }

    fn is_dangling(&self, e: &Self::Edge) -> bool {
        unsafe { &*e.node.get() }.is_none()
    }

    fn get_target_node<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node {
        unsafe { &*e.node.get() }.as_ref().unwrap()
    }
}

impl<O, I> Node<O, I> {
    fn new(outcome: O, e: Vec<Edge<O, I>>) -> Self {
        Node {
            lock: RawMutex::INIT,
            internal: UnsafeCell::new(Internal::new(outcome)),
            edges: e,
        }
    }
}

impl<O, A> Deref for Edge<O, OnlyAction<A>> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.data.action
    }
}

impl<O, A> Deref for Edge<O, ActionWithStaticPolicy<A>> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.data.action
    }
}

impl<O, A> PriorPolicyStore for Edge<O, ActionWithStaticPolicy<A>> {
    fn prior_policy_score(&self) -> f32 {
        self.data.static_policy_score
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fmt::{Debug, Display};

    pub fn print_graph<H: Eq + Hash, O: Debug + Clone, D: Display>(
        ns: &SafeGraph<H, D, O>,
        n: &Node<O, D>,
        offset: u32,
        full: bool,
    ) {
        println!("node: {{internal: {:?}}}", unsafe { &*n.internal.get() });
        if n.selection_count() > 0 {
            for e in 0..ns.children_count(n) {
                for _ in 0..offset {
                    print!("  ");
                }
                print!("|-");
                let edge = ns.get_edge(n, e);
                print!("-> {{data: {}, internal: {:?}}} ", edge.data, unsafe {
                    &*edge.internal.get()
                });
                if !full || ns.is_dangling(edge) {
                    println!("-> !");
                } else {
                    print_graph(ns, ns.get_target_node(edge), offset + 1, full);
                }
            }
        }
    }
}

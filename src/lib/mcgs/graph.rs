

use parking_lot::lock_api::RawMutex;
use parking_lot::RawMutex as Mutex;
use std::cell::{UnsafeCell, RefCell};
use crate::lib::mcgs::common::Internal;
use std::sync::Arc;
use crate::lib::mcgs::search_graph::{OutcomeStore, SelectCountStore, ConcurrentAccess, SearchGraph, PriorPolicyStore};
use crate::lib::decision_process::SimpleMovingAverage;
use dashmap::DashMap;
use std::hash::Hash;
use std::ops::Deref;
use crate::lib::{OnlyAction, ActionWithStaticPolicy};

pub struct Node<O, I> {
    lock: Mutex,
    internal: UnsafeCell<Internal<O>>,
    edges: UnsafeCell<Vec<Edge<O, I>>>,
}

pub struct Edge<O, I> {
    data: I,
    internal: UnsafeCell<Internal<O>>,
    node: Arc<Node<O, I>>,
}

impl<O: Clone, I> Edge<O, I> {
    fn new(i: I, outcome: O, child: Arc<Node<O, I>>) -> Self {
        Edge {
            data: i,
            internal: UnsafeCell::new(Internal::new(outcome.clone())),
            node: child.clone(),
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

    fn mark_solved(&self) {
        unsafe { &mut *self.internal.get() }.mark_solved()
    }
}

impl<I, O: Clone + SimpleMovingAverage> OutcomeStore<O> for Edge<O, I> {
    fn expected_outcome(&self) -> O {
        unsafe { &*self.internal.get() }.expected_sample.clone()
    }

    fn is_solved(&self) -> bool {
        panic!("Edge doesn't support solution storing")
    }

    fn add_sample(&self, outcome: &O, weight: u32) {
        unsafe { &mut *self.internal.get() }.add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.sample_count
    }

    fn mark_solved(&self) {
        panic!("Edge doesn't support solution storing")
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
    nodes: DashMap<H, Arc<Node<O, D>>>
}

impl<H: Eq + Hash, D, O> SafeGraph<H, D, O> {
    fn new(outcome: O) -> Self {
        // TODO: set capacity
        SafeGraph {
            default_outcome: outcome,
            nodes: DashMap::new()
        }
    }
}

pub trait Keyable<K> {
    fn key(&self) -> K;
}

impl<H: Eq + Hash, S: Keyable<H>, O:Clone, D> SearchGraph<D, S> for SafeGraph<H, D, O> {
    type Node = Node<O, D>;
    type Edge = Edge<O, D>;
    type NodeRef = Arc<Self::Node>;

    fn create_node(&self, s: &S) -> Self::NodeRef {
        let key = s.key();
        let opt = self.nodes.get(&key);
        if opt.is_none() {
            let n = Arc::new(Node::new(self.default_outcome.clone()));
            self.nodes.insert(key, n.clone()).unwrap();
            n
        } else {
            opt.unwrap().clone()
        }
    }

    fn clear(&self, _: Self::NodeRef) {
        self.nodes.clear()
    }

    fn is_leaf(&self, n: &Self::Node) -> bool {
        unsafe { &*n.edges.get() }.is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        unsafe { &*n.edges.get() }.len() as u32
    }

    fn create_children<L: Iterator<Item=D>>(&self, n: &Self::Node, l: L) {
        unimplemented!()
    }

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge {
        unsafe {
            let r = unsafe { &*n.edges.get() };
            r.get_unchecked(ix as usize)
        }
    }

    fn get_target_node<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node {
        &e.node
    }
}

impl<O, I> Node<O, I> {
    fn new(outcome: O) -> Self {
        Node {
            lock: RawMutex::INIT,
            internal: UnsafeCell::new(Internal::new(outcome)),
            edges: UnsafeCell::new(vec![]),
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
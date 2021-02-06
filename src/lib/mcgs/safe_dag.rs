use crate::lib::decision_process::Outcome;
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph};
use atomic_float::AtomicF32;
use parking_lot::RwLock;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

struct Node<I> {
    expected_outcome: AtomicF32,
    selection_count: AtomicU32,
    sample_count: AtomicU32,
    is_solved: AtomicBool,
    edges: RwLock<Vec<Edge<I>>>,
}

struct Edge<I> {
    data: I,
    expected_outcome: AtomicF32,
    selection_count: AtomicU32,
    sample_count: AtomicU32,
    node: Box<Node<I>>,
}

// TODO: see if ordering can be relaxed
impl<I> OutcomeStore<f32> for Node<I> {
    fn expected_outcome(&self) -> f32 {
        self.expected_outcome.load(Ordering::SeqCst)
    }

    fn is_solved(&self) -> bool {
        self.is_solved.load(Ordering::SeqCst)
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        self.sample_count.fetch_add(weight, Ordering::SeqCst);
        self.expected_outcome.fetch_add(*outcome, Ordering::SeqCst);
    }

    fn sample_count(&self) -> u32 {
        self.sample_count.load(Ordering::SeqCst)
    }

    fn mark_solved(&self) {
        self.is_solved.store(true, Ordering::SeqCst)
    }
}

// TODO: see if ordering can be relaxed
impl<I> OutcomeStore<f32> for Edge<I> {
    fn expected_outcome(&self) -> f32 {
        self.expected_outcome.load(Ordering::SeqCst)
    }

    fn is_solved(&self) -> bool {
        panic!("Edge doesn't support solution storing")
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        self.sample_count.fetch_add(weight, Ordering::SeqCst);
        self.expected_outcome.fetch_add(*outcome, Ordering::SeqCst);
    }

    fn sample_count(&self) -> u32 {
        self.sample_count.load(Ordering::SeqCst)
    }

    fn mark_solved(&self) {
        panic!("Edge doesn't support solution storing")
    }
}

struct SafeDag<O> {
    phantom: PhantomData<O>,
}

impl<O> SearchGraph for SafeDag<O> {
    type Node = Node<O>;
    type Edge = Edge<O>;

    fn create_node<'a>(&self) -> &'a mut Self::Node {
        unsafe { &mut *Box::into_raw(Box::new(Node::new())) }
    }

    fn drop_node(&self, n: &mut Self::Node) {
        unsafe {
            Box::from_raw(n);
        }
    }

    fn is_leaf(&self, n: &Self::Node) -> bool {
        n.edges.read().is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        n.edges.read().len() as u32
    }

    fn create_children<I: Into<Self::Edge>, L: Iterator<Item = I>>(&self, n: &Self::Node, l: L) {
        unimplemented!()
    }

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge {
        unimplemented!()
    }

    fn get_target<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node {
        e.node.as_ref()
    }
}

impl<I> Node<I> {
    fn new() -> Self {
        Node {
            expected_outcome: Default::default(),
            selection_count: Default::default(),
            sample_count: Default::default(),
            is_solved: Default::default(),
            edges: Default::default(),
        }
    }
}

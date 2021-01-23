use crate::lib::mcts::node_store::NodeStore;
use atomic_float::AtomicF32;
use num::FromPrimitive;
use std::cell::{RefCell, UnsafeCell};
use std::ops::{Deref, Range};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::marker::PhantomData;

pub struct ThreadSafeNodeStore<I>(PhantomData<I>);

impl<I> ThreadSafeNodeStore<I> {
    pub(crate) fn new() -> Self {
        ThreadSafeNodeStore(PhantomData)
    }
}

impl<I> NodeStore for ThreadSafeNodeStore<I> {
    type Node = Node<I>;
    type Edge = Edge<I>;
    type EdgeRef = u32;
    type NodeRef = NodePointer<I>;

    fn new_node(&self) -> Self::NodeRef {
        let mut result = NodePointer::new();
        result.create_data_if_missing();
        result
    }

    type EdgeIter = Range<u32>;

    fn edges_outgoing(&self, n: &Self::Node) -> Self::EdgeIter {
        0..(self.degree_outgoing(n) as u32)
    }

    fn degree_outgoing(&self, n: &Self::Node) -> usize {
        unsafe {&*n.children.get()}.len()
    }

    fn get_target_node(&self, e: &Self::Edge) -> Self::NodeRef {
        e.target_node.clone()
    }

    fn get_edge(&self, n: &Self::Node, e: Self::EdgeRef) -> &Self::Edge {
        unsafe {(&*n.children.get()).get_unchecked(e as usize)}
    }
}

pub struct Node<I> {
    total_selection_count: AtomicU32,

    accumulated_samples: AtomicF32,
    sample_count: AtomicU32,

    is_solved: AtomicBool,
    is_expanding: AtomicBool,
    is_finished_expanding: AtomicBool,
    is_started_adding_children: AtomicBool,

    children: UnsafeCell<Vec<Edge<I>>>,
}

pub struct Edge<I> {
    data: I,
    selection_count: AtomicU32,
    target_node: NodePointer<I>,
}

impl<I> Deref for Edge<I> {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

pub(crate) struct OnlyAction<A> {
    action: A
}

impl<A> Deref for OnlyAction<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

struct ActionWithStaticPolicy<A> {
    action: A,
    static_policy_score: f32
}

impl<A> Deref for ActionWithStaticPolicy<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

pub struct NodePointer<A> {
    data: Option<Arc<Node<A>>>,
}

impl<A> NodePointer<A> {
    fn new() -> Self {
        NodePointer { data: None }
    }

    fn create_data_if_missing(&mut self) {
        if self.data.is_none() {
            self.data.replace(Arc::new(Node::new()));
        }
    }
}

impl<A> Node<A> {
    fn new() -> Self {
        Node {
            total_selection_count: AtomicU32::new(0),
            accumulated_samples: AtomicF32::new(0.0),
            sample_count: AtomicU32::new(0),
            is_solved: AtomicBool::new(false),
            is_expanding: AtomicBool::new(false),
            is_finished_expanding: AtomicBool::new(false),
            is_started_adding_children: AtomicBool::new(false),
            children: UnsafeCell::new(vec![]),
        }
    }
}

impl<A> crate::lib::mcts::node_store::Node<f32> for Node<A> {
    fn has_started_expanding(&self) -> bool {
        self.is_expanding.swap(true, Ordering::SeqCst)
    }

    fn finish_expanding(&self) {
        self.is_finished_expanding.store(true, Ordering::SeqCst)
    }

    fn is_finished_expanding(&self) -> bool {
        self.is_finished_expanding.load(Ordering::SeqCst)
    }

    fn add_sample(&self, r: f32, w: u32) {
        unimplemented!()
    }

    fn expectation(&self) -> f32 {
        let cnt = self.sample_count.load(Ordering::SeqCst);
        if cnt == 0 {
            f32::MAX
        } else {
            self.accumulated_samples.load(Ordering::SeqCst) / f32::from_u32(cnt).unwrap()
        }
    }

    fn is_solved(&self) -> bool {
        self.is_solved.load(Ordering::SeqCst)
    }

    fn mark_solved(&self) {
        self.is_solved.store(true, Ordering::SeqCst)
    }
}

impl<A> crate::lib::mcts::node_store::Edge<f32> for Edge<A> {
    fn selection_count(&self) -> u32 {
        self.selection_count.load(Ordering::SeqCst)
    }

    fn increment_selection_count(&self) {
        self.selection_count.fetch_add(1, Ordering::SeqCst);
    }

    fn expected_sample(&self) -> f32 {
        unimplemented!()
    }
}


#[derive(Clone, Copy)]
struct EdgeRef(u32);

impl<A> Deref for NodePointer<A> {
    type Target = Node<A>;

    fn deref(&self) -> &Self::Target {
        self.data.as_ref().unwrap()
    }
}

impl<A> Clone for NodePointer<A> {
    fn clone(&self) -> Self {
        NodePointer {
            data: self.data.clone(),
        }
    }
}

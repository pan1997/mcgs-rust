use crate::lib::mcts::node_store::{
    ActionWithStaticPolicy, EdgeWithStaticData, NodeStore, OnlyAction,
};
use atomic_float::AtomicF32;
use num::FromPrimitive;
use std::cell::{Cell, UnsafeCell};
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

// TODO: complete this...

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
    type NodeRef = NodeRef<I>;

    fn new_node(&self) -> Self::NodeRef {
        let mut result = NodeRef::new();
        result.create_data_if_missing();
        result
    }

    fn create_outgoing_edges<T: Into<Self::Edge>>(
        &self,
        n: &Self::Node,
        iter: impl Iterator<Item = T>,
    ) {
        if !n.has_started_expanding() {
            let edges = unsafe { &mut *n.children.get() };
            edges.extend(iter.map(|e| e.into()));
            n.is_finished_expanding.store(true, Ordering::Release);
        }
    }

    type EdgeIter = Range<u32>;

    fn edges_outgoing(&self, n: &Self::Node) -> Self::EdgeIter {
        0..(self.degree_outgoing(n) as u32)
    }

    fn degree_outgoing(&self, n: &Self::Node) -> usize {
        unsafe { &*n.children.get() }.len()
    }

    fn get_target_node(&self, e: &Self::Edge) -> Self::NodeRef {
        e.target_node.clone()
    }

    fn get_or_create_target_node(&self, e: &mut Self::Edge) -> Self::NodeRef {
        e.target_node.create_data_if_missing();
        e.target_node.clone()
    }

    fn get_edge(&self, n: &Self::Node, e: Self::EdgeRef) -> &Self::Edge {
        unsafe { (&*n.children.get()).get_unchecked(e as usize) }
    }

    fn get_edge_mut(&self, n: &Self::Node, e: Self::EdgeRef) -> &mut Self::Edge {
        unsafe { (&mut *n.children.get()).get_unchecked_mut(e as usize) }
    }
}

pub struct Node<I> {
    total_selection_count: AtomicU32,

    // TODO: merge these two into a single atomic tuple
    accumulated_samples: AtomicF32,
    sample_count: AtomicU32,

    // Using a cell instead of a atomic bool because this is not a major issue
    // TODO: verify this
    is_solved: Cell<bool>,
    is_expanding: AtomicBool,
    is_finished_expanding: AtomicBool,

    children: UnsafeCell<Vec<Edge<I>>>,
}

pub struct Edge<I> {
    data: I,
    selection_count: AtomicU32,
    target_node: NodeRef<I>,
}

impl<I> Deref for Edge<I> {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<I> From<I> for Edge<I> {
    fn from(i: I) -> Self {
        Edge {
            data: i,
            selection_count: AtomicU32::new(0),
            target_node: NodeRef::new(),
        }
    }
}

pub struct NodeRef<A> {
    data: Option<Arc<Node<A>>>,
}

impl<A> NodeRef<A> {
    fn new() -> Self {
        NodeRef { data: None }
    }

    fn create_data_if_missing(&mut self) {
        if self.is_dangling() {
            self.data.replace(Arc::new(Node::new()));
        }
    }

    fn is_dangling(&self) -> bool {
        self.data.is_none()
    }
}

impl<A> Node<A> {
    fn new() -> Self {
        Node {
            total_selection_count: AtomicU32::new(0),
            accumulated_samples: AtomicF32::new(0.0),
            sample_count: AtomicU32::new(0),
            is_solved: Cell::new(false),
            is_expanding: AtomicBool::new(false),
            is_finished_expanding: AtomicBool::new(false),
            children: UnsafeCell::new(vec![]),
        }
    }

    fn expected_score(&self) -> f32 {
        let cnt = self.sample_count.load(Ordering::SeqCst);
        if cnt == 0 {
            f32::MAX
        } else {
            self.accumulated_samples.load(Ordering::SeqCst) / f32::from_u32(cnt).unwrap()
        }
    }

    fn has_started_expanding(&self) -> bool {
        self.is_expanding.swap(true, Ordering::SeqCst)
    }
}

impl<A> crate::lib::mcts::node_store::Node<f32> for Node<A> {
    fn finish_expanding(&self) {
        self.is_finished_expanding.store(true, Ordering::SeqCst)
    }

    fn is_finished_expanding(&self) -> bool {
        self.is_finished_expanding.load(Ordering::SeqCst)
    }

    fn add_sample(&self, r: f32, w: u32) {
        self.sample_count.fetch_add(w, Ordering::SeqCst);
        self.accumulated_samples.fetch_add(r, Ordering::SeqCst);
    }

    fn expectation(&self) -> f32 {
        self.expected_score()
    }

    fn is_solved(&self) -> bool {
        self.is_solved.get()
    }

    fn mark_solved(&self) {
        self.is_solved.set(true)
    }

    fn increment_selection_count(&self) {
        self.total_selection_count.fetch_add(1, Ordering::SeqCst);
    }

    fn total_selection_count(&self) -> u32 {
        self.total_selection_count.load(Ordering::SeqCst)
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
        if self.target_node.is_dangling() {
            f32::MAX
        } else {
            self.target_node.expected_score()
        }
    }
}

impl<A> EdgeWithStaticData<f32> for Edge<ActionWithStaticPolicy<A>> {
    fn get_static_data(&self) -> f32 {
        self.data.static_policy_score
    }
}

#[derive(Clone, Copy)]
struct EdgeRef(u32);

impl<A> Deref for NodeRef<A> {
    type Target = Node<A>;

    fn deref(&self) -> &Self::Target {
        &self.data.as_ref().unwrap()
    }
}

impl<A> Clone for NodeRef<A> {
    fn clone(&self) -> Self {
        NodeRef {
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::lib::mcts::node_store::Edge;
    use std::fmt::Display;

    pub(crate) fn print_tree<I>(ns: &ThreadSafeNodeStore<I>, n: &Node<I>)
    where
        I: Display,
    {
        fn print_tree_inner<I>(ns: &ThreadSafeNodeStore<I>, n: &Node<I>, d: u32)
        where
            I: Display,
        {
            println!(
                "node: {{s_count: {}, n_count: {}, score: {:.2e}}}",
                n.total_selection_count.load(Ordering::SeqCst),
                n.sample_count.load(Ordering::SeqCst),
                n.expected_score()
            );
            for e in ns.edges_outgoing(n) {
                for _ in 0..d {
                    print!("|-");
                }
                let edge = ns.get_edge(n, e);
                print!(
                    "-> {{data: {}, s_count: {}, score: {:.2e}}} ",
                    edge.data,
                    edge.selection_count.load(Ordering::SeqCst),
                    edge.expected_sample()
                );
                // TODO: display child
                let c = edge.target_node.clone();
                if c.is_dangling() {
                    println!("-> !");
                } else {
                    print_tree_inner(ns, &c, d + 1);
                }
            }
        }
        print_tree_inner(ns, n, 0);
    }
}

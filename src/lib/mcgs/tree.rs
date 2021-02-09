use crate::lib::decision_process::SimpleMovingAverage;
use crate::lib::mcgs::common::Internal;
use crate::lib::mcgs::samples::Samples;
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use parking_lot::lock_api::RawMutex;
use parking_lot::RawMutex as Mutex;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::Deref;

pub struct Node<O, I> {
    lock: Mutex,
    internal: UnsafeCell<Internal<O>>,
    edges: UnsafeCell<Vec<Edge<O, I>>>,
}

pub struct Edge<O, I> {
    data: I,
    internal: UnsafeCell<Internal<O>>,
    node: Node<O, I>,
}

impl<O: Clone, I> Edge<O, I> {
    fn new(i: I, outcome: O) -> Self {
        Edge {
            data: i,
            internal: UnsafeCell::new(Internal::new(outcome.clone())),
            node: Node::new(outcome),
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

pub struct SafeTree<D, O> {
    default_outcome: O,
    phantom: PhantomData<(D, O)>,
}

impl<D, O> SafeTree<D, O> {
    pub(crate) fn new(outcome: O) -> Self {
        SafeTree {
            default_outcome: outcome,
            phantom: PhantomData,
        }
    }
}

impl<S, O: Clone, D> SearchGraph<D, S> for SafeTree<D, O> {
    type Node = Node<O, D>;
    type Edge = Edge<O, D>;
    type NodeRef = Box<Self::Node>;

    fn create_node(&self, _: &S) -> Self::NodeRef {
        Box::new(Node::new(self.default_outcome.clone()))
    }

    fn clear(&self, _: Self::NodeRef) {
        // no extra op needed
    }

    fn is_leaf(&self, n: &Self::Node) -> bool {
        unsafe { &*n.edges.get() }.is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        unsafe { &*n.edges.get() }.len() as u32
    }

    fn create_children<L: Iterator<Item = D>>(&self, n: &Self::Node, l: L) {
        let children = unsafe { &mut *n.edges.get() };
        if children.is_empty() {
            for e in l {
                children.push(Self::Edge::new(e, self.default_outcome.clone()));
            }
        }
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fmt::{Debug, Display};

    #[test]
    fn t1() {
        let x = Samples::new();
        for i in 0..5 {
            println!("{}", x);
            x.add_sample(i as f32, 1);
        }
        println!("{}", x);
        x.mark_solved();
        for i in 0..5 {
            println!("{}", x);
            x.add_sample(i as f32, 1);
        }
        println!("{}", x);
    }

    pub fn print_graph<O: Debug + Clone, D: Display>(
        ns: &SafeTree<D, O>,
        n: &Node<O, D>,
        offset: u32,
        full: bool,
    ) {
        println!(
            "node: {{s_count: {}, score: {:?}}}",
            n.selection_count(),
            unsafe { &*n.internal.get() }
        );
        if n.selection_count() > 0 {
            for e in 0..SearchGraph::<_, ()>::children_count(ns, n) {
                for _ in 0..offset {
                    print!("  ");
                }
                print!("|-");
                let edge = SearchGraph::<_, ()>::get_edge(ns, n, e);
                print!(
                    "-> {{data: {}, s_count: {}}} ",
                    edge.data,
                    edge.selection_count(),
                    //Edge::<D>::expected_outcome(edge)
                );
                if !full {
                    println!("-> !");
                } else {
                    print_graph(
                        ns,
                        SearchGraph::<_, ()>::get_target_node(ns, edge),
                        offset + 1,
                        full,
                    );
                }
            }
        }
    }
}
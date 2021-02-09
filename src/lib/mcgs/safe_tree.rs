use crate::lib::mcgs::samples::Samples;
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use parking_lot::lock_api::RawMutex;
use parking_lot::RawMutex as Mutex;
use std::cell::UnsafeCell;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

pub struct Node<I> {
    lock: Mutex,
    internal: UnsafeCell<Internal<f32>>,
    edges: UnsafeCell<Vec<Edge<I>>>,
}

pub struct Edge<I> {
    data: I,
    internal: UnsafeCell<Internal<f32>>,
    node: Node<I>,
}

unsafe impl<I> Sync for Node<I> {}

impl<I> OutcomeStore<f32> for Node<I> {
    fn expected_outcome(&self) -> f32 {
        unsafe { &*self.internal.get() }.expected_sample
    }

    fn is_solved(&self) -> bool {
        unsafe { &*self.internal.get() }.is_solved()
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        unsafe { &mut *self.internal.get() }.add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.sample_count
    }

    fn mark_solved(&self) {
        unsafe { &mut *self.internal.get() }.mark_solved()
    }
}

impl<I> OutcomeStore<f32> for Edge<I> {
    fn expected_outcome(&self) -> f32 {
        unsafe { &*self.internal.get() }.expected_sample
    }

    fn is_solved(&self) -> bool {
        panic!("Edge doesn't support solution storing")
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        unsafe { &mut *self.internal.get() }.add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.sample_count
    }

    fn mark_solved(&self) {
        panic!("Edge doesn't support solution storing")
    }
}

impl<I> SelectCountStore for Node<I> {
    fn selection_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.selection_count
    }

    fn increment_selection_count(&self) {
        unsafe { &mut *self.internal.get() }.selection_count += 1
    }
}

impl<I> SelectCountStore for Edge<I> {
    fn selection_count(&self) -> u32 {
        unsafe { &*self.internal.get() }.selection_count
    }

    fn increment_selection_count(&self) {
        unsafe { &mut *self.internal.get() }.selection_count += 1
    }
}

impl<I> ConcurrentAccess for Node<I> {
    fn lock(&self) {
        self.lock.lock()
    }

    fn unlock(&self) {
        unsafe { self.lock.unlock() }
    }
}

impl<I> ConcurrentAccess for Edge<I> {
    // Locks not needed as we use the locks from the node
    fn lock(&self) {}
    fn unlock(&self) {}
}

pub struct SafeTree<D, O> {
    phantom: PhantomData<(D, O)>,
}

impl<D, O> SafeTree<D, O> {
    pub(crate) fn new() -> Self {
        SafeTree {
            phantom: PhantomData,
        }
    }
}

impl<S, O, D> SearchGraph<S> for SafeTree<D, O> {
    type Node = Node<D>;
    type Edge = Edge<D>;

    fn create_node(&self, _: &S) -> Box<Self::Node> {
        Box::new(Node::new())
    }

    fn is_leaf(&self, n: &Self::Node) -> bool {
        unsafe { &*n.edges.get() }.is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        unsafe { &*n.edges.get() }.len() as u32
    }

    fn create_children<I: Into<Self::Edge>, L: Iterator<Item = I>>(&self, n: &Self::Node, l: L) {
        let children = unsafe { &mut *n.edges.get() };
        if children.is_empty() {
            for e in l {
                children.push(e.into());
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

impl<I> Node<I> {
    fn new() -> Self {
        Node {
            lock: RawMutex::INIT,
            internal: UnsafeCell::new(Internal::new()),
            edges: UnsafeCell::new(vec![]),
        }
    }
}

impl<I> OutcomeStore<Vec<f32>> for Node<I> {
    fn expected_outcome(&self) -> Vec<f32> {
        unsafe { vec![(*self.internal.get()).expected_sample] }
    }

    fn is_solved(&self) -> bool {
        unsafe { (*self.internal.get()).is_solved() }
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        unsafe { (*self.internal.get()).add_sample(&outcome[0], weight) }
    }

    fn sample_count(&self) -> u32 {
        unsafe { (*self.internal.get()).sample_count }
    }

    fn mark_solved(&self) {
        unsafe { (*self.internal.get()).mark_solved() }
    }
}

impl<I> OutcomeStore<Vec<f32>> for Edge<I> {
    fn expected_outcome(&self) -> Vec<f32> {
        unsafe { vec![(*self.internal.get()).expected_sample] }
    }

    fn is_solved(&self) -> bool {
        unsafe { (*self.internal.get()).is_solved() }
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        unsafe { (*self.internal.get()).add_sample(&outcome[0], weight) }
    }

    fn sample_count(&self) -> u32 {
        unsafe { (*self.internal.get()).sample_count }
    }

    fn mark_solved(&self) {
        unsafe { (*self.internal.get()).mark_solved() }
    }
}

impl<A> Deref for Edge<OnlyAction<A>> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.data.action
    }
}

impl<A> Deref for Edge<ActionWithStaticPolicy<A>> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.data.action
    }
}

impl<A> PriorPolicyStore for Edge<ActionWithStaticPolicy<A>> {
    fn prior_policy_score(&self) -> f32 {
        self.data.static_policy_score
    }
}

impl<I> From<I> for Edge<I> {
    fn from(data: I) -> Self {
        Edge {
            data,
            internal: UnsafeCell::new(Internal::new()),
            node: Node::new(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crossbeam::atomic::AtomicCell;
    use std::fmt::Display;

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

    pub fn print_graph<O, D: Display>(ns: &SafeTree<D, O>, n: &Node<D>, offset: u32, full: bool) {
        println!(
            "node: {{s_count: {}, score: {}}}",
            n.selection_count(),
            unsafe { &*n.internal.get() }
        );
        if n.selection_count() >= 0 {
            for e in 0..SearchGraph::<()>::children_count(ns, n) {
                for _ in 0..offset {
                    print!("  ");
                }
                print!("|-");
                let edge = SearchGraph::<()>::get_edge(ns, n, e);
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
                        SearchGraph::<()>::get_target_node(ns, edge),
                        offset + 1,
                        full,
                    );
                }
            }
        }
    }
}

struct Internal<O> {
    expected_sample: O,
    sample_count: u32,
    selection_count: u32,
}

impl<O: Default> Internal<O> {
    fn new() -> Self {
        Internal {
            expected_sample: Default::default(),
            sample_count: 0,
            selection_count: 0,
        }
    }

    fn is_solved(&self) -> bool {
        self.sample_count == u32::MAX
    }

    fn mark_solved(&mut self) {
        self.sample_count = u32::MAX
    }
}

impl Internal<f32> {
    fn add_sample(&mut self, x: &f32, weight: u32) {
        if self.is_solved() {
            //println!("attempting to add a sample to a solved store. ignoring")
        } else {
            self.sample_count += weight;
            self.expected_sample +=
                (weight as f32) * (x - self.expected_sample) / (self.sample_count as f32)
        }
    }
}

impl<O: Display + Default> Display for Internal<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{n_count: {}, score: {:.2}",
            self.selection_count, self.expected_sample
        )?;
        if self.is_solved() {
            write!(f, "}}")
        } else {
            write!(f, ", count: {}", self.sample_count)
        }
    }
}

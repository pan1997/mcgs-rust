use crate::lib::decision_process::{Outcome, SimpleMovingAverage};
use crate::lib::mcgs::samples::Samples;
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use parking_lot::lock_api::RawMutex;
use parking_lot::RawMutex as Mutex;
use std::cell::UnsafeCell;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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

impl<I> OutcomeStore<f32> for Node<f32, I> {
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

impl<I> OutcomeStore<f32> for Edge<f32, I> {
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

    fn create_node(&self, _: &S) -> Box<Self::Node> {
        Box::new(Node::new(self.default_outcome.clone()))
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

impl<I> OutcomeStore<Vec<f32>> for Node<Vec<f32>, I> {
    fn expected_outcome(&self) -> Vec<f32> {
        unsafe { (&*self.internal.get()).expected_sample.clone() }
    }

    fn is_solved(&self) -> bool {
        unsafe { (*self.internal.get()).is_solved() }
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        unsafe { (&mut *self.internal.get()).add_sample(&outcome, weight) }
    }

    fn sample_count(&self) -> u32 {
        unsafe { (*self.internal.get()).sample_count }
    }

    fn mark_solved(&self) {
        unsafe { (*self.internal.get()).mark_solved() }
    }
}

impl<I> OutcomeStore<Vec<f32>> for Edge<Vec<f32>, I> {
    fn expected_outcome(&self) -> Vec<f32> {
        unsafe { (&*self.internal.get()).expected_sample.clone() }
    }

    fn is_solved(&self) -> bool {
        unsafe { (*self.internal.get()).is_solved() }
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        unsafe { (&mut *self.internal.get()).add_sample(&outcome, weight) }
    }

    fn sample_count(&self) -> u32 {
        unsafe { (*self.internal.get()).sample_count }
    }

    fn mark_solved(&self) {
        unsafe { (*self.internal.get()).mark_solved() }
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

struct Internal<O> {
    expected_sample: O,
    sample_count: u32,
    selection_count: u32,
}

impl<O> Internal<O> {
    fn new(outcome: O) -> Self {
        Internal {
            expected_sample: outcome,
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

impl<O: SimpleMovingAverage> Internal<O> {
    fn add_sample(&mut self, x: &O, weight: u32) {
        if !self.is_solved() {
            self.sample_count += weight;
            self.expected_sample
                .update_with_moving_average(x, weight, self.sample_count)
        }
    }
}
/*
// TODO: generalize
impl Internal<f32> {
    fn add_sample(&mut self, x: &f32, weight: u32) {
        if !self.is_solved() {
            self.sample_count += weight;
            self.expected_sample +=
                (weight as f32) * (x - self.expected_sample) / (self.sample_count as f32)
        }
    }
}
impl Internal<Vec<f32>> {
    fn add_sample(&mut self, x: &Vec<f32>, weight: u32) {
        if !self.is_solved() {
            self.sample_count += weight;
            for index in 0..self.expected_sample.len() {
                self.expected_sample[index] += (weight as f32)
                    * (x[index] - self.expected_sample[index])
                    / (self.sample_count as f32)
            }
        }
    }
}
*/
impl<O: Display> Display for Internal<O> {
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

impl<O: Debug> Debug for Internal<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{n_count: {}, score: {:.2?}",
            self.selection_count, self.expected_sample
        )?;
        if self.is_solved() {
            write!(f, "}}")
        } else {
            write!(f, ", count: {}", self.sample_count)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crossbeam::atomic::AtomicCell;
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

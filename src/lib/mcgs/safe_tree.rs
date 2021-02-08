use crate::lib::mcgs::samples::Samples;
use crate::lib::mcgs::search_graph::{
    OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use parking_lot::RwLock;

// TODO: switch to a single raw mutex and cells
pub struct Node<I> {
    selection_count: AtomicU32,
    samples: Samples,
    edges: RwLock<Vec<Edge<I>>>,
}

pub struct Edge<I> {
    data: I,
    selection_count: AtomicU32,
    samples: Samples,

    node: Node<I>,
}

unsafe impl<I> Sync for Node<I> {}

impl<I> OutcomeStore<f32> for Node<I> {
    fn expected_outcome(&self) -> f32 {
        self.samples.expected_sample()
    }

    fn is_solved(&self) -> bool {
        self.samples.is_solved()
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        self.samples.add_sample(*outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        self.samples.count()
    }

    fn mark_solved(&self) {
        self.samples.mark_solved()
    }
}

impl<I> OutcomeStore<f32> for Edge<I> {
    fn expected_outcome(&self) -> f32 {
        self.samples.expected_sample()
    }

    fn is_solved(&self) -> bool {
        panic!("Edge doesn't support solution storing")
    }

    fn add_sample(&self, outcome: &f32, weight: u32) {
        self.samples.add_sample(*outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        self.samples.count()
    }

    fn mark_solved(&self) {
        panic!("Edge doesn't support solution storing")
    }
}

// TODO: see if ordering can be relaxed
impl<I> SelectCountStore for Node<I> {
    fn selection_count(&self) -> u32 {
        self.selection_count.load(Ordering::SeqCst)
    }

    fn increment_selection_count(&self) {
        self.selection_count.fetch_add(1, Ordering::SeqCst);
    }
}

// TODO: see if ordering can be relaxed
impl<I> SelectCountStore for Edge<I> {
    fn selection_count(&self) -> u32 {
        self.selection_count.load(Ordering::SeqCst)
    }

    fn increment_selection_count(&self) {
        self.selection_count.fetch_add(1, Ordering::SeqCst);
    }
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
        n.edges.read().is_empty()
    }

    fn children_count(&self, n: &Self::Node) -> u32 {
        n.edges.read().len() as u32
    }

    fn create_children<I: Into<Self::Edge>, L: Iterator<Item = I>>(&self, n: &Self::Node, l: L) {
        let mut children = n.edges.write();
        for e in l {
            children.push(e.into());
        }
    }

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge {
        unsafe {
            let r = n.edges.read();
            let child = r.get_unchecked(ix as usize);

            // ( ⚆ _ ⚆ )
            // allow borrow outside the guard
            std::mem::transmute(child)
        }
    }


    fn get_target_node<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node {
        &e.node
    }
}

impl<I> Node<I> {
    fn new() -> Self {
        Node {
            selection_count: AtomicU32::new(0),
            samples: Samples::new(),
            edges: RwLock::new(vec![]),
        }
    }
}

impl<I> OutcomeStore<Vec<f32>> for Node<I> {
    fn expected_outcome(&self) -> Vec<f32> {
        vec![self.samples.expected_sample()]
    }

    fn is_solved(&self) -> bool {
        self.samples.is_solved()
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        //debug_assert!(outcome.len() == 1);
        self.samples.add_sample(outcome[0], weight)
    }

    fn sample_count(&self) -> u32 {
        self.samples.count()
    }

    fn mark_solved(&self) {
        self.samples.mark_solved()
    }
}

impl<I> OutcomeStore<Vec<f32>> for Edge<I> {
    fn expected_outcome(&self) -> Vec<f32> {
        vec![self.samples.expected_sample()]
    }

    fn is_solved(&self) -> bool {
        panic!("Edge doesn't support solution storing")
    }

    fn add_sample(&self, outcome: &Vec<f32>, weight: u32) {
        //debug_assert!(outcome.len() == 1);
        self.samples.add_sample(outcome[0], weight)
    }

    fn sample_count(&self) -> u32 {
        self.samples.count()
    }

    fn mark_solved(&self) {
        panic!("Edge doesn't support solution storing")
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
            selection_count: AtomicU32::new(0),
            samples: Samples::new(),
            node: Node::new()
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fmt::Display;
    use crossbeam::atomic::AtomicCell;

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
            n.selection_count.load(Ordering::SeqCst),
            n.samples,
        );
        if n.selection_count.load(Ordering::SeqCst) >= 0 {
            for e in 0..SearchGraph::<()>::children_count(ns, n) {
                for _ in 0..offset {
                    print!("  ");
                }
                print!("|-");
                let edge = SearchGraph::<()>::get_edge(ns, n, e);
                print!(
                    "-> {{data: {}, s_count: {}, score: {}}} ",
                    edge.data,
                    edge.selection_count.load(Ordering::SeqCst),
                    edge.samples
                );
                if !full {
                    println!("-> !");
                } else {
                    print_graph(ns, SearchGraph::<()>::get_target_node(ns, edge), offset + 1,
                                full);
                }
            }
        }
    }
}

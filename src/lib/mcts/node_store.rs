use petgraph::visit::EdgeRef;
use std::ops::{AddAssign, Deref};

trait NodeStore {
    type Node;
    type Edge;

    // TODO: see if we can avoid EdgeRef here
    type EdgeRef: Clone + Deref<Target=Self::Node>;
    type NodeRef: Clone + Deref<Target=Self::Edge>;

    fn new_node(&self) -> Self::NodeRef;

    type EdgeIter: Iterator<Item=Self::EdgeRef>;
    fn edges_outgoing(&self, n: &Self::Node) -> Self::EdgeIter;
    /// This is the length of the above iterator
    fn degree_outgoing(&self, n: &Self::Node) -> usize;

    fn get_target_node(&self, e: Self::Edge) -> Self::NodeRef;
}


trait Node<R: AddAssign> {
    fn add_sample(&self, r: R, w: u32);
    fn expectation(&self) -> R;

    fn is_solved(&self) -> bool;
    // Once a node is marked solved, it is expected that we do not add more samples. This might
    // or might not be enforced by the node store
    fn mark_solved(&self);
}

trait Edge<A> {
    fn selection_count(&self) -> u32;
    fn increment_selection_count(&self);
}
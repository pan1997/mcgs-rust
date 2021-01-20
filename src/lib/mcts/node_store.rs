use std::ops::{AddAssign, Deref};

pub trait NodeStore {
    type Node;
    type Edge;

    // TODO: see if we can avoid EdgeRef here
    type EdgeRef: Clone + Deref<Target = Self::Node>;
    type NodeRef: Clone + Deref<Target = Self::Edge>;

    fn new_node(&self) -> Self::NodeRef;

    type EdgeIter: Iterator<Item = Self::EdgeRef>;
    fn edges_outgoing(&self, n: &Self::Node) -> Self::EdgeIter;
    /// This is the length of the above iterator
    fn degree_outgoing(&self, n: &Self::Node) -> usize;

    fn get_target_node(&self, e: Self::Edge) -> Self::NodeRef;
}

trait Node<R: AddAssign> {
    // This returns true if this is the first call to start expanding across all threads.
    fn start_expanding(&self) -> bool;
    // This is to be called after all outgoing edges of this node have been created
    fn finish_expanding(&self);

    // TODO: do we need an additional state?
    // We cannot and should not descend a child if there exists a child that does not have a
    // sample, which will be taken care of by the property of expected_sample, which returns
    // R::max_value when sample count is 0

    fn add_sample(&self, r: R, w: u32);
    fn expectation(&self) -> R;

    fn is_solved(&self) -> bool;
    // Once a node is marked solved, it is expected that we do not add more samples. This might
    // or might not be enforced by the node store
    fn mark_solved(&self);
}

trait Edge<R> {
    fn selection_count(&self) -> u32;
    fn increment_selection_count(&self);

    // This is supposed to return the expected sample of the node pointed byt this edge. Note
    // that this is defined to be R::max_value() when the node has not been expanded yet
    fn expected_sample(&self) -> R;
}

pub trait TreePolicy<NS: NodeStore> {
    fn sample_edge(&self, store: &NS, n: &NS::Node, depth: u32) -> Option<NS::EdgeRef>;
}

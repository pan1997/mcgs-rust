use rand::prelude::IteratorRandom;
use std::fmt::{Display, Formatter};
use std::ops::Deref;

pub trait NodeStore {
    type Node;
    type Edge;

    // TODO: see if we can avoid EdgeRef here
    type EdgeRef: Clone;
    type NodeRef: Clone + Deref<Target = Self::Node> + Nullable;

    fn new_node(&self) -> Self::NodeRef;
    fn create_outgoing_edges<I: Into<Self::Edge>>(
        &self,
        n: &Self::Node,
        iter: impl Iterator<Item = I>,
    );

    type EdgeIter: Iterator<Item = Self::EdgeRef>;
    fn edges_outgoing(&self, n: &Self::Node) -> Self::EdgeIter;
    /// This is the length of the above iterator
    fn degree_outgoing(&self, n: &Self::Node) -> usize;

    fn get_target_node(&self, e: &Self::Edge) -> Self::NodeRef;
    // This always returns a NodeRef to a legal node, as it creates a new node if the pointer was
    // nil before. Note that this still has a race condition in the current ThreadSafeNodeStore.
    fn get_or_create_target_node(&self, e: &mut Self::Edge) -> Self::NodeRef;

    // TODO: see if this can be avoidied by EdgeRef: Clone + Deref<Self::Edge>
    fn get_edge(&self, n: &Self::Node, e: Self::EdgeRef) -> &Self::Edge;
    fn get_edge_mut(&self, n: &Self::Node, e: Self::EdgeRef) -> &mut Self::Edge;
}

pub trait Nullable {
    fn is_some(&self) -> bool;
}

pub(crate) trait Node<R> {
    // This returns true for all but the first call across all threads
    //fn has_started_expanding(&self) -> bool;
    // This is to be called after all outgoing edges of this node have been created
    fn finish_expanding(&self);

    fn is_finished_expanding(&self) -> bool;

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

    // TODO: see if this can be avoided or merged with the edge's increment call.
    fn increment_selection_count(&self);

    fn total_selection_count(&self) -> u32;
}

pub(crate) trait Edge<R> {
    fn selection_count(&self) -> u32;
    fn increment_selection_count(&self);

    // This is supposed to return the expected sample of the node pointed byt this edge. Note
    // that this is defined to be R::max_value() when the node has not been expanded yet
    fn expected_sample(&self) -> R;
}

pub(crate) trait EdgeWithStaticData<I> {
    fn get_static_data(&self) -> I;
}

pub trait TreePolicy<NS: NodeStore> {
    fn sample_edge(&self, store: &NS, n: &NS::Node, depth: u32) -> Option<NS::EdgeRef>;
}

pub(crate) struct OnlyAction<A> {
    pub(crate) action: A,
}

impl<A> Display for OnlyAction<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{action: {}}}", self.action)
    }
}

impl<A> Deref for OnlyAction<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

pub(crate) struct ActionWithStaticPolicy<A> {
    pub(crate) action: A,
    pub(crate) static_policy_score: f32,
}

impl<A> Display for ActionWithStaticPolicy<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{action: {}, static_score: {}}}",
            self.action, self.static_policy_score
        )
    }
}

impl<A> Deref for ActionWithStaticPolicy<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

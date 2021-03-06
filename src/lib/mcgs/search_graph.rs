use std::ops::Deref;

pub trait OutcomeStore<O> {
    fn expected_outcome(&self) -> O;
    fn is_solved(&self) -> bool;

    // Need to use atomics/mutexes
    fn add_sample(&self, outcome: &O, weight: u32);
    fn sample_count(&self) -> u32;
    fn mark_solved(&self, outcome: &O);
}

pub trait SelectCountStore {
    fn selection_count(&self) -> u32;

    // Need to use atomics/mutexes
    fn increment_selection_count(&self);
}

pub trait ConcurrentAccess {
    fn lock(&self);
    fn unlock(&self);
}

pub trait PriorPolicyStore {
    fn prior_policy_score(&self) -> f32;
}

pub trait SearchGraph<D, H> {
    type Node;
    type Edge;
    type NodeRef: Deref<Target = Self::Node>;

    fn create_node<L: Iterator<Item = D>>(&self, h: H, l: L) -> Self::NodeRef;
    fn add_child<'a, L: Iterator<Item = D>>(&self, h: H, e: &'a Self::Edge, l: L)
        -> &'a Self::Node;
    fn link_child(&self, node: &Self::NodeRef, e: &Self::Edge);
    fn clear(&self, n: Self::NodeRef);

    fn is_leaf(&self, n: &Self::Node) -> bool;
    fn children_count(&self, n: &Self::Node) -> u32;
    // need some kind of structural lock on the node (or a atomic state)
    //fn create_children<L: Iterator<Item = D>>(&self, n: &Self::Node, l: L);

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge;
    fn is_dangling(&self, e: &Self::Edge) -> bool;
    fn get_target_node<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node;
}

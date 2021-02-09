pub trait OutcomeStore<O> {
    fn expected_outcome(&self) -> O;
    fn is_solved(&self) -> bool;

    // Need to use atomics/mutexes
    fn add_sample(&self, outcome: &O, weight: u32);
    fn sample_count(&self) -> u32;
    fn mark_solved(&self);
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

pub trait SearchGraph<S> {
    type Node;
    type Edge;

    fn create_node(&self, s: &S) -> Box<Self::Node>;

    fn is_leaf(&self, n: &Self::Node) -> bool;
    fn children_count(&self, n: &Self::Node) -> u32;
    // need some kind of structural lock on the node (or a atomic state)
    fn create_children<I: Into<Self::Edge>, L: Iterator<Item = I>>(&self, n: &Self::Node, l: L);

    fn get_edge<'a>(&self, n: &'a Self::Node, ix: u32) -> &'a Self::Edge;
    fn get_target_node<'a>(&self, e: &'a Self::Edge) -> &'a Self::Node;
}

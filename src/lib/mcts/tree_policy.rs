use crate::lib::mcts::node_store::{Edge, Node, NodeStore, TreePolicy};
use rand::prelude::IteratorRandom;
use rand::{thread_rng, Rng};

pub(crate) struct RandomTreePolicy;
impl<NS: NodeStore> TreePolicy<NS> for RandomTreePolicy {
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        _: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        store.edges_outgoing(n).choose(&mut rand::thread_rng())
    }
}

struct FirstNonVistedTreePolicy;
impl<NS: NodeStore> TreePolicy<NS> for FirstNonVistedTreePolicy
where
    NS::Edge: Edge<f32>,
{
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        _: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        debug_assert!(store.degree_outgoing(n) > 0);
        for edge_ref in store.edges_outgoing(n) {
            let edge = store.get_edge(n, edge_ref.clone());
            if edge.selection_count() == 0 {
                return Some(edge_ref);
            }
        }
        // Return a random edge when all edges have been visited
        RandomTreePolicy.sample_edge(store, n, 0)
    }
}

pub(crate) struct UctTreePolicy {
    exploration_constant: f32,
}

impl UctTreePolicy {
    pub(crate) fn new(f: f32) -> Self {
        assert!(f > 0.0);
        UctTreePolicy {
            exploration_constant: f,
        }
    }
}

// The default is to return the first edge with 0 selection count.
impl<NS: NodeStore> TreePolicy<NS> for UctTreePolicy
where
    NS::Edge: Edge<f32>,
    NS::Node: Node<f32>,
{
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        _: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        let degree_outgoing = store.degree_outgoing(n) as u32;
        debug_assert!(degree_outgoing > 0);
        let selection_count = n.total_selection_count();
        if selection_count < degree_outgoing {
            FirstNonVistedTreePolicy.sample_edge(store, n, 0)
        } else {
            let factor = (selection_count as f32).ln();
            generic_tree_policy(
                store,
                n,
                |edge_selection_count| {
                    self.exploration_constant * (factor / (edge_selection_count as f32)).sqrt()
                },
                selection_count
            )
        }
    }
}


pub(crate) struct PuctTreePolicy {
    exploration_constant: f32,
}

impl PuctTreePolicy {
    pub(crate) fn new(f: f32) -> Self {
        assert!(f > 0.0);
        PuctTreePolicy {
            exploration_constant: f,
        }
    }
}

// The default is to return the first edge with 0 selection count.
impl<NS: NodeStore> TreePolicy<NS> for PuctTreePolicy
    where
        NS::Edge: Edge<f32>,
        NS::Node: Node<f32>,
{
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        _: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        let degree_outgoing = store.degree_outgoing(n) as u32;
        debug_assert!(degree_outgoing > 0);
        let selection_count = n.total_selection_count();
        if selection_count < degree_outgoing {
            FirstNonVistedTreePolicy.sample_edge(store, n, 0)
        } else {
            let factor = (selection_count as f32).sqrt();
            generic_tree_policy(
                store,
                n,
                |edge_selection_count| {
                    self.exploration_constant * factor / ((1 + edge_selection_count) as f32).sqrt()
                },
                selection_count
            )
        }
    }
}


fn generic_tree_policy<NS: NodeStore, F: Fn(u32) -> f32>(
    store: &NS,
    n: &NS::Node,
    f: F,
    selection_count: u32
) -> Option<NS::EdgeRef>
where
    NS::Edge: Edge<f32>,
    NS::Node: Node<f32>,
{
    let mut best_edge = None;
    let mut best_edge_total_score = f32::MIN;
    // This is used to denote the number of edges that have the same score as the best edge
    let mut best_edge_equivalence_count = 0;
    let factor = (selection_count as f32).ln();
    // let mut best_edge_exploration_score = f32::MIN;
    // This is the number of edges that have the best score
    // let mut best_score_count = 0;
    for edge_ref in store.edges_outgoing(n) {
        let edge = store.get_edge(n, edge_ref.clone());
        let edge_selection_count = edge.selection_count();

        // Return the first non selected edge. This is still needed because of possible
        // race conditions and to handle cases when we switch between tree policies. Can
        // be removed later when it is ensured that this is not needed.
        if edge_selection_count == 0 {
            return Some(edge_ref);
        }

        let edge_expected_reward = edge.expected_sample();
        let edge_exploration_score = f(edge_selection_count);
        let total_score = edge_expected_reward + edge_exploration_score;

        if total_score > best_edge_total_score {
            best_edge_total_score = total_score;
            best_edge = Some(edge_ref);
            best_edge_equivalence_count = 1;
        } else if total_score == best_edge_total_score {
            best_edge_equivalence_count += 1;
            if rand::thread_rng().gen_range(0..best_edge_equivalence_count) == 0 {
                best_edge = Some(edge_ref);
            }
        }
    }
    best_edge
}

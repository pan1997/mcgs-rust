use crate::lib::mcts::node_store::{Edge, EdgeWithStaticData, Node, NodeStore, TreePolicy};
use rand::prelude::IteratorRandom;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Dirichlet;

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

// TODO: find a way to define this for all I
pub(crate) struct MostVisitedTreePolicy;
impl<NS: NodeStore> TreePolicy<NS> for MostVisitedTreePolicy
where
    NS::Edge: Edge<f32>,
{
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        depth: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        debug_assert!(store.degree_outgoing(n) > 0);
        let mut best_edge = None;
        let mut best_edge_count = 0;
        for edge_ref in store.edges_outgoing(n) {
            let edge = store.get_edge(n, edge_ref.clone());
            let count = edge.selection_count();
            if count > best_edge_count {
                best_edge = Some(edge_ref);
                best_edge_count = count;
            }
        }
        best_edge
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
                |edge_selection_count, _| {
                    self.exploration_constant * (factor / (edge_selection_count as f32)).sqrt()
                },
                selection_count,
                true,
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
    NS::Edge: Edge<f32> + EdgeWithStaticData<f32>,
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
        let factor = (selection_count as f32).sqrt();
        generic_tree_policy(
            store,
            n,
            |edge_selection_count, edge| {
                self.exploration_constant * edge.get_static_data() * factor
                    / ((1 + edge_selection_count) as f32).sqrt()
            },
            selection_count,
            false,
        )
    }
}

pub(crate) struct PuctWithDiricheletTreePolicy {
    exploration_constant: f32,
    dirichelet_alpha: f32,
    dirichelet_weight: f32,
}

impl PuctWithDiricheletTreePolicy {
    pub(crate) fn new(f: f32, dirichelet_alpha: f32, dirichelet_weight: f32) -> Self {
        assert!(f > 0.0);
        PuctWithDiricheletTreePolicy {
            exploration_constant: f,
            dirichelet_alpha,
            dirichelet_weight,
        }
    }

    fn dirichelet_noise(&self, l: usize) -> Vec<f32> {
        let l = Dirichlet::new_with_size(self.dirichelet_alpha, l).unwrap();
        l.sample(&mut rand::thread_rng())
    }
}
// The default is to return the first edge with 0 selection count.
impl<NS: NodeStore> TreePolicy<NS> for PuctWithDiricheletTreePolicy
where
    NS::Edge: Edge<f32> + EdgeWithStaticData<f32>,
    NS::Node: Node<f32>,
{
    fn sample_edge(
        &self,
        store: &NS,
        n: &<NS as NodeStore>::Node,
        _: u32,
    ) -> Option<<NS as NodeStore>::EdgeRef> {
        let degree_outgoing = store.degree_outgoing(n);
        debug_assert!(degree_outgoing > 0);
        if degree_outgoing == 1 {
            store.edges_outgoing(n).next()
        } else {
            let selection_count = n.total_selection_count();
            let factor = (selection_count as f32).sqrt();
            let noise = self.dirichelet_noise(degree_outgoing);
            let mut index = 0;
            generic_tree_policy(
                store,
                n,
                |edge_selection_count, edge| {
                    let result = self.exploration_constant
                        * (self.dirichelet_weight * noise[index]
                            + (1.0 - self.dirichelet_weight) * edge.get_static_data())
                        * factor
                        / ((1 + edge_selection_count) as f32).sqrt();
                    index += 1;
                    result
                },
                selection_count,
                false,
            )
        }
    }
}

fn generic_tree_policy<NS: NodeStore, F: FnMut(u32, &NS::Edge) -> f32>(
    store: &NS,
    n: &NS::Node,
    mut f: F,
    selection_count: u32,
    enable_shorting: bool,
) -> Option<NS::EdgeRef>
where
    NS::Edge: Edge<f32>,
    NS::Node: Node<f32>,
{
    let mut best_edge = None;
    let mut best_edge_total_score = f32::MIN;
    let mut best_edge_exploration_score = f32::MIN;
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
        if enable_shorting && edge_selection_count == 0 {
            return Some(edge_ref);
        }

        let edge_expected_reward = edge.expected_sample();
        let edge_exploration_score = f(edge_selection_count, edge);
        let total_score = edge_expected_reward + edge_exploration_score;

        if total_score > best_edge_total_score {
            best_edge_total_score = total_score;
            best_edge_exploration_score = edge_exploration_score;
            best_edge = Some(edge_ref);
            best_edge_equivalence_count = 1;
        } else if total_score == best_edge_total_score {
            if edge_exploration_score > best_edge_exploration_score {
                best_edge_total_score = total_score;
                best_edge_exploration_score = edge_exploration_score;
                best_edge = Some(edge_ref);
                best_edge_equivalence_count = 1;
            } else if edge_exploration_score == best_edge_exploration_score {
                best_edge_equivalence_count += 1;
                if rand::thread_rng().gen_range(0..best_edge_equivalence_count) == 0 {
                    best_edge = Some(edge_ref);
                }
            }
        }
    }
    best_edge
}

use crate::lib::decision_process::{DecisionProcess, Outcome};
use crate::lib::mcgs::search_graph::{
    OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use num::ToPrimitive;
use rand::{thread_rng, Rng};
use std::ops::Deref;

pub(crate) trait SelectionPolicy<P, G>
where
    G: SearchGraph,
    P: DecisionProcess,
{
    //TODO: see if we can return the edge ref
    fn select(&self, problem: &P, store: &G, node: &G::Node, agent: P::Agent, depth: u32) -> u32;
}

struct RandomPolicy;
impl<P: DecisionProcess, G: SearchGraph> SelectionPolicy<P, G> for RandomPolicy {
    fn select(&self, _: &P, store: &G, node: &G::Node, _: P::Agent, _: u32) -> u32 {
        let k = store.children_count(node);
        debug_assert!(k > 0);
        thread_rng().gen_range(0..k)
    }
}

struct FirstNonVisitedPolicy;
impl<P, G> SelectionPolicy<P, G> for FirstNonVisitedPolicy
where
    P: DecisionProcess,
    G: SearchGraph,
    G::Edge: SelectCountStore,
{
    fn select(&self, _: &P, store: &G, node: &G::Node, _: P::Agent, _: u32) -> u32 {
        let edge_count = store.children_count(node);
        debug_assert!(edge_count > 0);
        for edge_index in 0..edge_count {
            if store.get_edge(node, edge_index).selection_count() == 0 {
                return edge_index;
            }
        }
        // Returns an invalid edge when all edges have been visited
        // TODO: do we need to switch to random for this case?
        edge_count
    }
}

struct UctPolicy {
    exploration_weight: f32,
}

impl<P, G> SelectionPolicy<P, G> for UctPolicy
where
    P: DecisionProcess,
    G: SearchGraph,
    G::Edge: SelectCountStore + OutcomeStore<P::Outcome>,
    G::Node: SelectCountStore,
    <P::Outcome as Outcome<P::Agent>>::RewardType: ToPrimitive,
{
    fn select(&self, _: &P, store: &G, node: &G::Node, agent: P::Agent, _: u32) -> u32 {
        let edge_count = store.children_count(node);
        debug_assert!(edge_count > 0);

        let mut best_edge_score = f32::MIN;
        let mut best_edge = edge_count;
        let c_lg = (node.selection_count() as f32).ln();

        for edge_index in 0..edge_count {
            let edge = store.get_edge(node, edge_index);
            let q = edge
                .expected_outcome()
                .reward_for_agent(agent)
                .to_f32()
                .unwrap();

            let edge_selection_count = edge.selection_count();
            if edge_selection_count == 0 {
                return edge_index;
            }

            let exploration_score = (c_lg / edge_selection_count as f32).sqrt();
            let score = q + self.exploration_weight * exploration_score;
            if score > best_edge_score {
                best_edge_score = score;
                best_edge = edge_index;
            }
        }
        // Returns an invalid edge when all edges have been visited
        // TODO: do we need to switch to random for this case?
        edge_count
    }
}

struct PuctPolicy {
    puct_init: f32,
    puct_base: f32,
}

impl<P, G> SelectionPolicy<P, G> for PuctPolicy
where
    P: DecisionProcess,
    G: SearchGraph,
    G::Edge: SelectCountStore + OutcomeStore<P::Outcome> + PriorPolicyStore,
    G::Node: SelectCountStore,
    <P::Outcome as Outcome<P::Agent>>::RewardType: ToPrimitive,
{
    fn select(&self, _: &P, store: &G, node: &G::Node, agent: P::Agent, _: u32) -> u32 {
        let edge_count = store.children_count(node);
        debug_assert!(edge_count > 0);

        let mut best_edge_score = f32::MIN;
        let mut best_edge = edge_count;
        let node_selection_count = node.selection_count() as f32;
        let c_sqrt = node_selection_count.sqrt();

        for edge_index in 0..edge_count {
            let edge = store.get_edge(node, edge_index);
            let q = edge
                .expected_outcome()
                .reward_for_agent(agent)
                .to_f32()
                .unwrap();

            let edge_selection_count = edge.selection_count();

            let p_uct = ((node_selection_count + self.puct_base + 1.0) / self.puct_base).ln();
            let exploration_score =
                p_uct * edge.prior_policy_score() * (c_sqrt / (1 + edge_selection_count) as f32);

            let score = q + exploration_score;

            if score > best_edge_score {
                best_edge_score = score;
                best_edge = edge_index;
            }
        }
        // Returns an invalid edge when all edges have been visited
        // TODO: do we need to switch to random for this case?
        edge_count
    }
}
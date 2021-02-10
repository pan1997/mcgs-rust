use crate::lib::decision_process::{DecisionProcess, Outcome};
use crate::lib::mcgs::search_graph::{
    OutcomeStore, PriorPolicyStore, SearchGraph, SelectCountStore,
};
use num::ToPrimitive;
use rand::{thread_rng, Rng};

pub trait SelectionPolicy<D, P, G, H>
where
    G: SearchGraph<D, H>,
    P: DecisionProcess,
{
    //TODO: see if we can return the edge ref
    fn select(&self, problem: &P, store: &G, node: &G::Node, agent: P::Agent, depth: u32) -> u32;
}

pub struct RandomPolicy;
impl<D, P: DecisionProcess, G: SearchGraph<D, H>, H> SelectionPolicy<D, P, G, H> for RandomPolicy {
    fn select(&self, _: &P, store: &G, node: &G::Node, _: P::Agent, _: u32) -> u32 {
        let k = store.children_count(node);
        debug_assert!(k > 0);
        thread_rng().gen_range(0..k)
    }
}

struct FirstNonVisitedPolicy;
impl<D, P, G, H> SelectionPolicy<D, P, G, H> for FirstNonVisitedPolicy
where
    P: DecisionProcess,
    G: SearchGraph<D, H>,
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
        edge_count
    }
}

pub struct MostVisitedPolicy;
impl MostVisitedPolicy {
    pub(crate) fn select<D, P, G, H>(&self, _: &P, store: &G, node: &G::Node) -> u32
    where
        P: DecisionProcess,
        G: SearchGraph<D, H>,
        G::Edge: SelectCountStore,
    {
        let edge_count = store.children_count(node);
        debug_assert!(edge_count > 0);
        let mut max_count = 0;
        let mut best_edge = 0;
        for edge_index in 0..edge_count {
            let edge = store.get_edge(node, edge_index);
            if edge.selection_count() > max_count {
                best_edge = edge_index;
                max_count = edge.selection_count()
            }
        }
        best_edge
    }
}

pub struct UctPolicy {
    exploration_weight: f32,
}

impl UctPolicy {
    pub fn new(w: f32) -> Self {
        UctPolicy {
            exploration_weight: w,
        }
    }
}

impl<D, P, G, H> SelectionPolicy<D, P, G, H> for UctPolicy
where
    P: DecisionProcess,
    G: SearchGraph<D, H>,
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
        best_edge
    }
}

pub struct PuctPolicy {
    puct_init: f32,
    puct_base: f32,
}

impl PuctPolicy {
    pub fn new(base: f32, init: f32) -> Self {
        PuctPolicy {
            puct_base: base,
            puct_init: init,
        }
    }
}

impl<D, P, G, H> SelectionPolicy<D, P, G, H> for PuctPolicy
where
    P: DecisionProcess,
    G: SearchGraph<D, H>,
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

            let p_uct = self.puct_init
                + ((node_selection_count + self.puct_base + 1.0) / self.puct_base).ln();
            let exploration_score =
                p_uct * edge.prior_policy_score() * (c_sqrt / (1 + edge_selection_count) as f32);

            let score = q + exploration_score;

            if score > best_edge_score {
                best_edge_score = score;
                best_edge = edge_index;
            }
        }

        best_edge
    }
}

struct WeightedRandomPolicy<P1, P2> {
    p_epsilon: P1,
    p: P2,
    epsilon: f32,
}

impl<D, P, G, P1, P2, H> SelectionPolicy<D, P, G, H> for WeightedRandomPolicy<P1, P2>
where
    G: SearchGraph<D, H>,
    P: DecisionProcess,
    P1: SelectionPolicy<D, P, G, H>,
    P2: SelectionPolicy<D, P, G, H>,
{
    fn select(&self, problem: &P, store: &G, node: &G::Node, agent: P::Agent, depth: u32) -> u32 {
        let w: f32 = thread_rng().gen();
        if w < self.epsilon {
            self.p_epsilon.select(problem, store, node, agent, depth)
        } else {
            self.p.select(problem, store, node, agent, depth)
        }
    }
}

pub struct WeightedRandomPolicyWithExpDepth<P1, P2> {
    p_epsilon: P1,
    p: P2,
    epsilon: f32,
    factor: f32,
}
impl<P1, P2> WeightedRandomPolicyWithExpDepth<P1, P2> {
    pub(crate) fn new(p1: P1, p2: P2, e: f32, f: f32) -> Self {
        WeightedRandomPolicyWithExpDepth {
            p: p2,
            p_epsilon: p1,
            epsilon: e,
            factor: f,
        }
    }
}
impl<D, P, G, P1, P2, H> SelectionPolicy<D, P, G, H> for WeightedRandomPolicyWithExpDepth<P1, P2>
where
    G: SearchGraph<D, H>,
    P: DecisionProcess,
    P1: SelectionPolicy<D, P, G, H>,
    P2: SelectionPolicy<D, P, G, H>,
{
    fn select(&self, problem: &P, store: &G, node: &G::Node, agent: P::Agent, depth: u32) -> u32 {
        let w: f32 = thread_rng().gen();
        let depth_factor = (self.factor * depth as f32).exp();
        if w < self.epsilon * depth_factor {
            self.p_epsilon.select(problem, store, node, agent, depth)
        } else {
            self.p.select(problem, store, node, agent, depth)
        }
    }
}

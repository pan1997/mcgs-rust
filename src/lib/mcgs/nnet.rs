use crate::lib::decision_process::c4::C4;
use crate::lib::decision_process::{DecisionProcess, Distance, Outcome, SimpleMovingAverage};
use crate::lib::mcgs::expansion_traits::ExpansionTrait;
use crate::lib::mcgs::graph::Hsh;
use crate::lib::mcgs::graph_policy::SelectionPolicy;
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, SearchGraph, SelectCountStore,
};
use crate::lib::mcgs::{ExtraPropagationTask, Search, TrajectoryPruner};
use rand::distributions::WeightedIndex;
use rand::{thread_rng, Rng};
use rand_distr::Distribution;
use std::ops::Deref;
use tch::nn::{Module, Path};
use tch::Tensor;

fn c4_get_search_entry<D, K, G>(
    p: &C4,
    g: &G,
    n: &G::Node,
    agent: <C4 as DecisionProcess>::Agent,
) -> (Tensor, Tensor, <C4 as DecisionProcess>::Action)
where
    G: SearchGraph<D, K>,
    G::Node: OutcomeStore<<C4 as DecisionProcess>::Outcome>,
    G::Edge: SelectCountStore + Deref<Target = <C4 as DecisionProcess>::Action>,
{
    let mut counts = vec![0.0f32; p.width()];
    for edge_index in 0..g.children_count(n) {
        let edge = g.get_edge(n, edge_index);
        let mv: &<C4 as DecisionProcess>::Action = edge;
        counts[mv.0 as usize] = edge.selection_count() as f32;
    }
    let total: f32 = counts.iter().sum();
    counts.iter_mut().for_each(|x| *x /= total);

    let score = n.expected_outcome().reward_for_agent(agent);
    let index = WeightedIndex::new(&counts)
        .unwrap()
        .sample(&mut thread_rng());
    return (
        Tensor::of_slice(&counts).unsqueeze(0),
        Tensor::from(score).unsqueeze(0),
        crate::lib::decision_process::c4::Move(index as u8),
    );
}

pub(crate) struct SelfPlayData {
    pub(crate) states_tensor: Tensor,
    pub(crate) policy_tensor: Tensor,
    pub(crate) value_tensor: Tensor,
}

pub(crate) fn append(tensor: Option<Tensor>, x: Tensor) -> Option<Tensor> {
    if tensor.is_none() {
        Some(x)
    } else {
        Some(Tensor::cat(&[tensor.unwrap(), x], 0))
    }
}

pub(crate) fn c4_generate_self_play_game<S, R, G, X, D, H, Z>(
    s: &Search<C4, S, R, G, X, D, H, Z>,
    count: u32,
) -> SelfPlayData
where
    G: SearchGraph<D, H::K>,
    S: SelectionPolicy<D, C4, G, H::K>,
    G::Node: SelectCountStore + OutcomeStore<<C4 as DecisionProcess>::Outcome> + ConcurrentAccess,
    G::Edge: SelectCountStore
        + OutcomeStore<<C4 as DecisionProcess>::Outcome>
        + Deref<Target = <C4 as DecisionProcess>::Action>,
    R: TrajectoryPruner<G::Edge, G::Node, <C4 as DecisionProcess>::Outcome>,
    H: Hsh<crate::lib::decision_process::c4::Board>,
    Z: ExtraPropagationTask<
        G,
        G::Node,
        <C4 as DecisionProcess>::Outcome,
        <C4 as DecisionProcess>::Agent,
    >,
    X: ExpansionTrait<C4, D, H::K>,
{
    let mut state = s.problem().start_state();
    let mut states_tensor = None;
    let mut policy_head = None;
    let mut value_head = None;

    while s.problem().is_finished(&state).is_none() {
        let n = s.get_new_node(&mut state);

        for _ in 0..count {
            s.one_iteration(&n, &mut state);
            if n.is_solved() {
                break;
            }
        }

        let (_policy_head, _value_head, mv) = c4_get_search_entry(
            s.problem(),
            s.search_graph(),
            &n,
            s.problem().agent_to_act(&state),
        );

        states_tensor = append(states_tensor, s.problem().generate_tensor(&state));
        policy_head = append(policy_head, _policy_head);
        value_head = append(value_head, _value_head);
        s.search_graph.clear(n);

        s.problem().transition(&mut state, &mv);
    }
    println!("{}", state);
    println!("{:?}", states_tensor.as_ref().unwrap().size());
    println!("{:?}", policy_head.as_ref().unwrap().size());
    println!("{:?}", value_head.as_ref().unwrap().size());
    SelfPlayData {
        states_tensor: states_tensor.unwrap(),
        policy_tensor: policy_head.unwrap(),
        value_tensor: value_head.unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::RandomSimulator;
    use crate::lib::mcgs::expansion_traits::BasicExpansion;
    use crate::lib::mcgs::graph::NoHash;
    use crate::lib::mcgs::graph_policy::UctPolicy;
    use crate::lib::mcgs::tree::SafeTree;
    use crate::lib::mcgs::AlwaysExpand;
    use crate::lib::OnlyAction;

    #[test]
    fn t1() {
        let s = Search::new(
            C4::new(9, 7),
            SafeTree::<OnlyAction<_>, _>::new(0.0),
            UctPolicy::new(2.4),
            BasicExpansion::new(RandomSimulator),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );
        let x = c4_generate_self_play_game(&s, 128);
        x.states_tensor.print();
        x.policy_tensor.print();
        x.value_tensor.print();
    }
}

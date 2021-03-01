use crate::lib::decision_process::c4::C4;
use crate::lib::decision_process::{DecisionProcess, Outcome};
use crate::lib::mcgs::expansion_traits::{BlockExpansionTrait, ExpansionTrait};
use crate::lib::mcgs::graph::Hsh;
use crate::lib::mcgs::graph_policy::SelectionPolicy;
use crate::lib::mcgs::nnet::{build_ffn, GNet, SelfPlayData};
use crate::lib::mcgs::search_graph::{
    ConcurrentAccess, OutcomeStore, SearchGraph, SelectCountStore,
};
use crate::lib::mcgs::{ExpansionResult, ExtraPropagationTask, Search, TrajectoryPruner};
use crate::lib::ActionWithStaticPolicy;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::ops::Deref;
use tch::nn::{ModuleT, Path};
use tch::{Device, IndexOp, Kind, Tensor};

pub fn net1(vs: &Path) -> GNet<impl ModuleT, impl ModuleT, impl ModuleT> {
    let kernel_sizes = [5, 5, 5, 5, 5, 5];
    let channel_counts = [3, 5, 10, 10, 10, 10, 10];
    let shared =
        super::build_sequential_cnn(vs, &kernel_sizes, &channel_counts).add_fn(|x| x.flat_view());
    let _value = build_ffn(vs, &[630, 50, 1]);
    let value = _value.add_fn(|x| 2 * x.sigmoid() - 1.0);
    let log_policy = build_ffn(vs, &[630, 100, 9]);
    GNet::new(shared, log_policy, value)
}

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
    let mut counts = vec![1.0f32; p.width()];
    // We need to normalise this as log_policy is not defined (or is -inf) otherwise
    for edge_index in 0..g.children_count(n) {
        let edge = g.get_edge(n, edge_index);
        let mv: &<C4 as DecisionProcess>::Action = edge;
        counts[mv.0 as usize] += edge.selection_count() as f32;
    }
    let total: f32 = counts.iter().sum();
    counts.iter_mut().for_each(|x| *x /= total);

    let score = n.expected_outcome().reward_for_agent(agent);
    let index = WeightedIndex::new(&counts)
        .unwrap()
        .sample(&mut thread_rng());
    return (
        Tensor::of_slice(&counts).unsqueeze(0),
        Tensor::from(score).unsqueeze(0).unsqueeze(0),
        crate::lib::decision_process::c4::Move(index as u8),
    );
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
    let mut states_tensors = vec![];
    let mut policy_heads = vec![];
    let mut value_heads = vec![];

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

        states_tensors.push(s.problem().generate_tensor(&state));
        policy_heads.push(_policy_head);
        value_heads.push(_value_head);
        s.search_graph.clear(n);

        s.problem().transition(&mut state, &mv);
    }
    SelfPlayData {
        states_tensor: Tensor::cat(&states_tensors, 0).to_kind(Kind::Float),
        policy_tensor: Tensor::cat(&policy_heads, 0),
        value_tensor: Tensor::cat(&value_heads, 0),
    }
}

impl<N1: ModuleT, N2: ModuleT, N3: ModuleT, K>
    BlockExpansionTrait<C4, ActionWithStaticPolicy<<C4 as DecisionProcess>::Action>, K>
    for GNet<N1, N2, N3>
{
    type OutputIter =
        <Vec<ActionWithStaticPolicy<<C4 as DecisionProcess>::Action>> as IntoIterator>::IntoIter;
    type Batch = (Vec<Tensor>, Vec<(K, <C4 as DecisionProcess>::Actions)>);

    fn accept(
        &self,
        problem: &C4,
        state: &mut <C4 as DecisionProcess>::State,
        state_key: K,
        batch: &mut Self::Batch,
    ) -> usize {
        batch.0.push(problem.generate_tensor(state));
        batch.1.push((state_key, problem.legal_actions(state)));
        batch.0.len() - 1
    }

    fn process_accepted(
        &self,
        batch: Self::Batch,
    ) -> Vec<ExpansionResult<<C4 as DecisionProcess>::Outcome, Self::OutputIter, K>> {
        let input = Tensor::cat(&batch.0, 0)
            .to_device(Device::Cuda(0))
            .to_kind(Kind::Float);
        let (log_policy, _value) = self.forward(&input);
        let value: Vec<f64> = _value.into();
        let policy: Vec<Vec<f32>> = log_policy.exp().into();
        let mut result = vec![];
        for (index, entry) in batch.1.into_iter().enumerate() {
            let mut edges: Vec<_> = entry
                .1
                .map(|m| ActionWithStaticPolicy {
                    action: m,
                    static_policy_score: policy[index][m.0 as usize],
                })
                .collect();
            let edges_sum: f32 = edges.iter().map(|e| e.static_policy_score).sum();
            edges
                .iter_mut()
                .for_each(|e| e.static_policy_score /= edges_sum);
            result.push(ExpansionResult {
                state_key: entry.0,
                outcome: value[index],
                edges: edges.into_iter(),
            })
        }
        result
    }

    fn new_batch(&self) -> Self::Batch {
        (vec![], vec![])
    }
}
impl<N1: ModuleT, N2: ModuleT, N3: ModuleT, K>
    ExpansionTrait<C4, ActionWithStaticPolicy<<C4 as DecisionProcess>::Action>, K>
    for GNet<N1, N2, N3>
{
    type OutputIter =
        <Vec<ActionWithStaticPolicy<<C4 as DecisionProcess>::Action>> as IntoIterator>::IntoIter;

    fn apply(
        &self,
        problem: &C4,
        state: &mut <C4 as DecisionProcess>::State,
        state_key: K,
    ) -> ExpansionResult<<C4 as DecisionProcess>::Outcome, Self::OutputIter, K> {
        let mut batch = self.new_batch();
        let _ = self.accept(problem, state, state_key, &mut batch);
        self.process_accepted(batch).into_iter().next().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::RandomSimulator;
    use crate::lib::mcgs::expansion_traits::BasicExpansion;
    use crate::lib::mcgs::graph::NoHash;
    use crate::lib::mcgs::graph_policy::{PuctPolicy, UctPolicy};
    use crate::lib::mcgs::nnet::process;
    use crate::lib::mcgs::tree::tests::print_tree;
    use crate::lib::mcgs::tree::SafeTree;
    use crate::lib::mcgs::AlwaysExpand;
    use crate::lib::OnlyAction;
    use std::path::Prefix::DeviceNS;
    use tch::nn::OptimizerConfig;
    use tch::{Device, Kind};

    #[test]
    fn t1fgg() {
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

        let vs = tch::nn::VarStore::new(Device::Cuda(0));
        let root = &vs.root();
        let mut nn = net1(root);

        let x = c4_generate_self_play_game(&s, 128);
        let c4 = s.problem();
        let state = c4.start_state();
        println!("{:?}", x.states_tensor.size());
        println!("{:?}", x.policy_tensor.size());
        println!("{:?}", x.value_tensor.size());
        let mut opt = tch::nn::Adam::default().build(&vs, 1e-2).unwrap();
        let before = nn.forward(
            &c4.generate_tensor(&state)
                .to_kind(Kind::Float)
                .to_device(Device::Cuda(0)),
        );
        process(&x, &mut nn, &mut opt, true);
        let after = nn.forward(
            &c4.generate_tensor(&state)
                .to_kind(Kind::Float)
                .to_device(Device::Cuda(0)),
        );

        before.1.print();
        after.1.print();
    }

    #[test]
    fn tsft() {
        let c4 = C4::new(9, 7);
        let state = c4.start_state();

        let vs = tch::nn::VarStore::new(Device::Cpu);
        let root = &vs.root();
        let nn = net1(root);
        //nn.forward(&c4.generate_tensor(&state).to_kind(Kind::Float));
    }

    #[test]
    fn t1fgggth() {
        let vs = tch::nn::VarStore::new(Device::Cuda(0));
        let root = &vs.root();

        let s = Search::new(
            C4::new(9, 7),
            SafeTree::<_, _>::new(0.0),
            PuctPolicy::new(20.0, 2.0),
            net1(root),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let c4 = s.problem();
        let mut state = c4.start_state();
        let n = s.get_new_node(&mut state);

        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..100 {
            s.one_iteration(&n, &mut state);
            print_tree(&s.search_graph, &n, 0, false);
        }
    }
}

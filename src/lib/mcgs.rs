mod expansion_traits;
mod graph_policy;
mod safe_dag;
mod search_graph;

use graph_policy::*;
use search_graph::*;

use crate::lib::decision_process::{DecisionProcess, Distance, Outcome, Simulator};
use crate::lib::mcgs::expansion_traits::ExpansionTrait;
use std::cmp::{max, min};
use std::ops::Deref;

pub struct ExpansionResult<O, EI> {
    outcome: O,
    prune: bool,
    edges: EI,
}

// TODO: update this doc
/// The first iteration of this is NOT Thread safe, as it is known that the bottleneck of
/// AlphaZero puct is actually the expansion phase which involves neural network evaluation. A
/// central thread/process is responsible for querying and updating the search tree based on work
/// performed by a set of expansion workers (that may or may not involve neural network
/// computation).
///
/// We also assume that the simulation phase is significantly more expensive than the select and
/// expand, so using a batched expansion running in parallel should be sufficient, (as in
/// https://arxiv.org/abs/1810.11755). This (almost universally true assumption) allows simpler
/// implementation and avoids expensive safeguards to enable safe sharing of the search tree
/// among multiple workers.

struct Search<P, S, R, G, X> {
    problem: P,
    search_graph: G,
    selection_policy: S,
    expand_operation: X,
    q_shorting_bound: R,
    maximum_trajectory_weight: u32,
}

enum SelectionResult<N, V> {
    Expand(N),
    Propogate(V, u32), // The second arg is the weight
}

impl<P, S, G, X>
    Search<P, S, <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType, G, X>
where
    P: DecisionProcess,
    G: SearchGraph<P::State>,
    S: SelectionPolicy<P, G>,
    G::Node: SelectCountStore + OutcomeStore<P::Outcome>,
    G::Edge: SelectCountStore + OutcomeStore<P::Outcome> + Deref<Target = P::Action>,
    <P::Outcome as Outcome<P::Agent>>::RewardType: Distance,
{
    fn new(
        problem: P,
        search_graph: G,
        selection_policy: S,
        expand_operation: X,
        q_shorting_bound: <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType,
        maximum_trajectory_weight: u32,
    ) -> Self {
        Search {
            problem,
            search_graph,
            selection_policy,
            expand_operation,
            q_shorting_bound,
            maximum_trajectory_weight,
        }
    }

    /// Either creates a new node and returns it for expansion, or returns a
    /// Propagate(outcome, weight). In both cases, the state and node correspond to the state
    /// after reaching the end of trajectory
    fn select<'a>(
        &self,
        root: &'a G::Node,
        state: &mut P::State,
        trajectory: &mut Vec<(&'a G::Node, &'a G::Edge)>,
        undo_stack: &mut Vec<P::UndoAction>,
    ) -> SelectionResult<&'a G::Node, P::Outcome> {
        let mut node = root;
        while !self.search_graph.is_leaf(node) {
            let agent = self.problem.agent_to_act(state);

            let edge = self.search_graph.get_edge(
                node,
                self.selection_policy
                    .select(&self.problem, &self.search_graph, node, agent, 0),
            );

            node.increment_selection_count();
            edge.increment_selection_count();

            trajectory.push((node, edge));
            undo_stack.push(self.problem.transition(state, &edge));

            // If this edge is dangling, we create a new node and then return it
            if self.search_graph.is_dangling(edge) {
                let next_node = self.search_graph.create_target(edge, state);
                return SelectionResult::Expand(next_node);
            }
            let next_node = self.search_graph.get_target(edge);

            // Check if this is a transposition (next_node has extra samples)
            let n_count = next_node.sample_count();
            let e_count = edge.sample_count();
            if n_count > e_count {
                let delta = next_node
                    .expected_outcome()
                    .reward_for_agent(agent)
                    .distance(&edge.expected_outcome().reward_for_agent(agent));
                if delta > self.q_shorting_bound {
                    return SelectionResult::Propogate(
                        next_node.expected_outcome(),
                        min(self.maximum_trajectory_weight, n_count - e_count),
                    );
                }
            }

            // We need an additional terminal check, as we can mark non leaf nodes as terminal
            if next_node.is_solved() {
                return SelectionResult::Propogate(next_node.expected_outcome(), 1);
            }

            node = next_node;
        }

        println!("hey.....");
        SelectionResult::Expand(node)
    }

    fn expand<I>(&self, node: &G::Node, state: &mut P::State) -> (P::Outcome, u32)
    where
        X: ExpansionTrait<P, I>,
        G::Edge: From<I>,
    {
        let expansion_result = self.expand_operation.apply(&self.problem, state);
        if expansion_result.prune {
            node.add_sample(&expansion_result.outcome, 1);
            node.mark_solved();
        } else {
            self.search_graph
                .create_children(node, expansion_result.edges);
        }
        (expansion_result.outcome, 1)
    }

    fn propagate(
        &self,
        trajectory: &mut Vec<(&G::Node, &G::Edge)>,
        outcome: P::Outcome,
        weight: u32,
    ) {
        // TODO: add graph propogation
        while let Some((node, edge)) = trajectory.pop() {
            edge.add_sample(&outcome, weight);
            node.add_sample(&outcome, weight);
        }
    }

    fn one_iteration<I>(&self, root: &G::Node, state: &mut P::State)
    where
        X: ExpansionTrait<P, I>,
        G::Edge: From<I>,
    {
        let trajectory = &mut vec![];
        let undo_stack = &mut vec![];
        let s = self.select(root, state, trajectory, undo_stack);

        let (outcome, weight) = match s {
            SelectionResult::Expand(n) => self.expand(n, state),
            SelectionResult::Propogate(outcome, weight) => (outcome, weight),
        };

        while let Some(u) = undo_stack.pop() {
            self.problem.undo_transition(state, u);
        }

        self.propagate(trajectory, outcome, weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::graph_dp::tests::{problem1, problem2, DSim};
    use crate::lib::decision_process::{DefaultSimulator, RandomSimulator};
    use crate::lib::mcgs::expansion_traits::{BasicExpansion, BasicExpansionWithUniformPrior};
    use crate::lib::mcgs::safe_dag::tests::print_graph;
    use crate::lib::mcgs::safe_dag::SafeDag;
    use petgraph::prelude::NodeIndex;
    use crate::lib::mcts::node_store::ActionWithStaticPolicy;

    #[test]
    fn random() {
        let s = Search::new(
            problem1(),
            SafeDag::<_, Vec<f32>>::new(),
            RandomPolicy,
            BasicExpansion::new(DefaultSimulator),
            0.01,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.search_graph.create_node(state);
        print_graph(&s.search_graph, n, 0);
        for _ in 0..10 {
            s.one_iteration(n, state);
            print_graph(&s.search_graph, n, 0);
        }
        SearchGraph::<NodeIndex>::drop_node(&s.search_graph, n);
    }

    #[test]
    fn uct() {
        let s = Search::new(
            problem1(),
            SafeDag::<_, Vec<f32>>::new(),
            UctPolicy::new(2.4),
            BasicExpansion::new(DefaultSimulator),
            0.01,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.search_graph.create_node(state);
        print_graph(&s.search_graph, n, 0);
        for _ in 0..100 {
            s.one_iteration(n, state);
            print_graph(&s.search_graph, n, 0);
        }
        SearchGraph::<NodeIndex>::drop_node(&s.search_graph, n);
    }

    #[test]
    fn puct_2p() {
        let s = Search::new(
            problem2(),
            SafeDag::<ActionWithStaticPolicy<u32>, Vec<f32>>::new(),
            PuctPolicy::new(2.0, 2.4),
            BasicExpansionWithUniformPrior::new(DSim),
            0.01,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.search_graph.create_node(state);
        print_graph(&s.search_graph, n, 0);
        for _ in 0..250 {
            s.one_iteration(n, state);
            print_graph(&s.search_graph, n, 0);
        }
        SearchGraph::<NodeIndex>::drop_node(&s.search_graph, n);
    }

    #[test]
    fn puct() {
        let s = Search::new(
            problem1(),
            SafeDag::<ActionWithStaticPolicy<u32>, Vec<f32>>::new(),
            PuctPolicy::new(2.0, 2.4),
            BasicExpansionWithUniformPrior::new(DefaultSimulator),
            0.01,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.search_graph.create_node(state);
        print_graph(&s.search_graph, n, 0);
        for _ in 0..100 {
            s.one_iteration(n, state);
            print_graph(&s.search_graph, n, 0);
        }
        SearchGraph::<NodeIndex>::drop_node(&s.search_graph, n);
    }
}

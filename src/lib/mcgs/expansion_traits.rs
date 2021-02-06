use crate::lib::decision_process::{DecisionProcess, Simulator};
use crate::lib::mcgs::SimulationResult;
use crate::lib::mcts::node_store::OnlyAction;

pub trait ExpansionTrait<P: DecisionProcess, I> {
    type OutputIter: Iterator<Item = I>;
    fn expand_and_simulate(
        &self,
        problem: &P,
        state: &mut P::State,
    ) -> SimulationResult<P::Outcome, Self::OutputIter>;
}

pub struct BasicExpansion<S> {
    simulator: S,
}

impl<S> BasicExpansion<S> {
    pub(crate) fn new(s: S) -> Self {
        BasicExpansion { simulator: s }
    }
}

impl<P: DecisionProcess, S> ExpansionTrait<P, OnlyAction<P::Action>> for BasicExpansion<S>
where
    S: Simulator<P>,
{
    type OutputIter = <Vec<OnlyAction<P::Action>> as IntoIterator>::IntoIter;

    fn expand_and_simulate(
        &self,
        problem: &P,
        state: &mut P::State,
    ) -> SimulationResult<<P as DecisionProcess>::Outcome, Self::OutputIter> {
        if let Some(outcome) = problem.is_finished(state) {
            SimulationResult {
                outcome,
                prune: true,
                edges: vec![].into_iter(),
            }
        } else {
            let edges: Vec<OnlyAction<P::Action>> = problem
                .legal_actions(state)
                .map(|action| OnlyAction { action })
                .collect();
            SimulationResult {
                outcome: self.simulator.sample_outcome(problem, state),
                prune: false,
                edges: edges.into_iter(),
            }
        }
    }
}

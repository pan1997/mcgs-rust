use crate::lib::decision_process::{DecisionProcess, Simulator};
use crate::lib::mcgs::ExpansionResult;
use crate::lib::mcts::node_store::{ActionWithStaticPolicy, OnlyAction};
use parking_lot::Mutex;

pub trait ExpansionTrait<P: DecisionProcess, I> {
    type OutputIter: Iterator<Item = I>;
    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
    ) -> ExpansionResult<P::Outcome, Self::OutputIter>;
}

pub trait BlockExpansionTrait<P: DecisionProcess, I> {
    type OutputIter: Iterator<Item = I>;
    fn accept(&self, problem: &P, state: &mut P::State) -> usize;
    fn process_accepted(&self) -> Vec<ExpansionResult<P::Outcome, Self::OutputIter>>;
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

    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
    ) -> ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter> {
        if let Some(outcome) = problem.is_finished(state) {
            ExpansionResult {
                outcome,
                prune: true,
                edges: vec![].into_iter(),
            }
        } else {
            let edges: Vec<OnlyAction<P::Action>> = problem
                .legal_actions(state)
                .map(|action| OnlyAction { action })
                .collect();
            ExpansionResult {
                outcome: self.simulator.sample_outcome(problem, state),
                prune: false,
                edges: edges.into_iter(),
            }
        }
    }
}

pub struct BasicExpansionWithUniformPrior<S> {
    simulator: S,
}
impl<S> BasicExpansionWithUniformPrior<S> {
    pub(crate) fn new(s: S) -> Self {
        BasicExpansionWithUniformPrior { simulator: s }
    }
}

impl<P: DecisionProcess, S> ExpansionTrait<P, ActionWithStaticPolicy<P::Action>>
    for BasicExpansionWithUniformPrior<S>
where
    S: Simulator<P>,
{
    type OutputIter = <Vec<ActionWithStaticPolicy<P::Action>> as IntoIterator>::IntoIter;

    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
    ) -> ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter> {
        if let Some(outcome) = problem.is_finished(state) {
            ExpansionResult {
                outcome,
                prune: true,
                edges: vec![].into_iter(),
            }
        } else {
            let actions: Vec<P::Action> = problem.legal_actions(state).collect();
            let static_policy_score = 1.0 / (actions.len() as f32);
            let edges: Vec<ActionWithStaticPolicy<P::Action>> = actions
                .into_iter()
                .map(|action| ActionWithStaticPolicy {
                    action,
                    static_policy_score,
                })
                .collect();
            ExpansionResult {
                outcome: self.simulator.sample_outcome(problem, state),
                prune: false,
                edges: edges.into_iter(),
            }
        }
    }
}

pub struct BlockExpansionFromBasic<X, S> {
    basic: X,
    queue: Mutex<Vec<S>>,
}

impl<X, S> BlockExpansionFromBasic<X, S> {
    pub fn new(x: X) -> Self {
        BlockExpansionFromBasic {
            basic: x,
            queue: Mutex::new(vec![]),
        }
    }
}

impl<P: DecisionProcess, X, I> BlockExpansionTrait<P, I>
    for BlockExpansionFromBasic<X, ExpansionResult<P::Outcome, X::OutputIter>>
where
    X: ExpansionTrait<P, I>,
    P::Outcome: Clone,
    ExpansionResult<P::Outcome, X::OutputIter>: Clone,
{
    type OutputIter = X::OutputIter;

    fn accept(&self, problem: &P, state: &mut <P as DecisionProcess>::State) -> usize {
        let mut q = self.queue.lock();
        q.push(self.basic.apply(problem, state));
        q.len() - 1
    }

    fn process_accepted(
        &self,
    ) -> Vec<ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter>> {
        let mut q = self.queue.lock();
        let result = q.clone();
        q.clear();
        result
    }
}

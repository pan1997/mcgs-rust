use crate::lib::decision_process::{DecisionProcess, Simulator};
use crate::lib::mcgs::ExpansionResult;
use crate::lib::{ActionWithStaticPolicy, OnlyAction};

pub trait ExpansionTrait<P: DecisionProcess, I, K> {
    type OutputIter: Iterator<Item = I>;
    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
        state_key: K,
    ) -> ExpansionResult<P::Outcome, Self::OutputIter, K>;
}

pub trait BlockExpansionTrait<P: DecisionProcess, I, K> {
    type OutputIter: Iterator<Item = I>;
    type Batch;
    fn accept(
        &self,
        problem: &P,
        state: &mut P::State,
        state_key: K,
        batch: &mut Self::Batch,
    ) -> usize;

    fn process_accepted(
        &self,
        batch: Self::Batch,
    ) -> Vec<ExpansionResult<P::Outcome, Self::OutputIter, K>>;

    fn new_batch(&self) -> Self::Batch;
}

pub struct BasicExpansion<S> {
    simulator: S,
}

impl<S> BasicExpansion<S> {
    pub(crate) fn new(s: S) -> Self {
        BasicExpansion { simulator: s }
    }
}

impl<P: DecisionProcess, S, K> ExpansionTrait<P, OnlyAction<P::Action>, K> for BasicExpansion<S>
where
    S: Simulator<P>,
{
    type OutputIter = <Vec<OnlyAction<P::Action>> as IntoIterator>::IntoIter;

    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
        state_key: K,
    ) -> ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter, K> {
        if let Some(outcome) = problem.is_finished(state) {
            ExpansionResult {
                outcome,
                edges: vec![].into_iter(),
                state_key,
            }
        } else {
            let edges: Vec<OnlyAction<P::Action>> = problem
                .legal_actions(state)
                .map(|action| OnlyAction { action })
                .collect();
            ExpansionResult {
                outcome: self.simulator.sample_outcome(problem, state),
                edges: edges.into_iter(),
                state_key,
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

impl<P: DecisionProcess, S, K> ExpansionTrait<P, ActionWithStaticPolicy<P::Action>, K>
    for BasicExpansionWithUniformPrior<S>
where
    S: Simulator<P>,
{
    type OutputIter = <Vec<ActionWithStaticPolicy<P::Action>> as IntoIterator>::IntoIter;

    fn apply(
        &self,
        problem: &P,
        state: &mut P::State,
        state_key: K,
    ) -> ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter, K> {
        if let Some(outcome) = problem.is_finished(state) {
            ExpansionResult {
                outcome,
                edges: vec![].into_iter(),
                state_key,
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
                edges: edges.into_iter(),
                state_key,
            }
        }
    }
}

pub struct BlockExpansionFromBasic<X> {
    basic: X,
}

impl<X> BlockExpansionFromBasic<X> {
    pub fn new(x: X) -> Self {
        BlockExpansionFromBasic { basic: x }
    }
}

impl<P: DecisionProcess, X, I, K> BlockExpansionTrait<P, I, K> for BlockExpansionFromBasic<X>
where
    X: ExpansionTrait<P, I, K>,
{
    type OutputIter = X::OutputIter;
    type Batch = Vec<ExpansionResult<P::Outcome, X::OutputIter, K>>;

    fn accept(
        &self,
        problem: &P,
        state: &mut <P as DecisionProcess>::State,
        state_key: K,
        batch: &mut Self::Batch,
    ) -> usize {
        batch.push(self.basic.apply(problem, state, state_key));
        batch.len() - 1
    }

    fn process_accepted(
        &self,
        batch: Self::Batch,
    ) -> Vec<ExpansionResult<P::Outcome, X::OutputIter, K>> {
        batch
    }

    fn new_batch(&self) -> Self::Batch {
        vec![]
    }
}

impl<P: DecisionProcess, X, I, K> ExpansionTrait<P, I, K> for BlockExpansionFromBasic<X>
where
    X: ExpansionTrait<P, I, K>,
{
    type OutputIter = X::OutputIter;

    fn apply(
        &self,
        problem: &P,
        state: &mut <P as DecisionProcess>::State,
        state_key: K,
    ) -> ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter, K> {
        self.basic.apply(problem, state, state_key)
    }
}

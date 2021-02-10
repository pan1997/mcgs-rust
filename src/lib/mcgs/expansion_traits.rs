use crate::lib::decision_process::{DecisionProcess, Simulator};
use crate::lib::mcgs::ExpansionResult;
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use parking_lot::Mutex;

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
    fn accept(&self, problem: &P, state: &mut P::State, state_key: K) -> usize;
    fn process_accepted(&self) -> Vec<ExpansionResult<P::Outcome, Self::OutputIter, K>>;
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

impl<P: DecisionProcess, X, I, K> BlockExpansionTrait<P, I, K>
    for BlockExpansionFromBasic<X, ExpansionResult<P::Outcome, X::OutputIter, K>>
where
    X: ExpansionTrait<P, I, K>,
    P::Outcome: Clone,
    ExpansionResult<P::Outcome, X::OutputIter, K>: Clone,
{
    type OutputIter = X::OutputIter;

    fn accept(
        &self,
        problem: &P,
        state: &mut <P as DecisionProcess>::State,
        state_key: K,
    ) -> usize {
        let mut q = self.queue.lock();
        q.push(self.basic.apply(problem, state, state_key));
        q.len() - 1
    }

    fn process_accepted(
        &self,
    ) -> Vec<ExpansionResult<<P as DecisionProcess>::Outcome, Self::OutputIter, K>> {
        let mut q = self.queue.lock();
        let result = q.clone();
        q.clear();
        result
    }
}

impl<P: DecisionProcess, X, I, K> ExpansionTrait<P, I, K>
    for BlockExpansionFromBasic<X, ExpansionResult<P::Outcome, X::OutputIter, K>>
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

use crate::lib::decision_process::DecisionProcess;
use std::fmt::{Display, Formatter};
use std::ops::Deref;

pub(crate) mod decision_process;
pub(crate) mod mcgs;
pub(crate) mod mcts;

pub(crate) struct OnlyAction<A> {
    pub(crate) action: A,
}

impl<A> Display for OnlyAction<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{action: {}}}", self.action)
    }
}

impl<A> Deref for OnlyAction<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

impl<A: Clone> Clone for OnlyAction<A> {
    fn clone(&self) -> Self {
        OnlyAction {
            action: self.action.clone(),
        }
    }
}

pub(crate) struct ActionWithStaticPolicy<A> {
    pub(crate) action: A,
    pub(crate) static_policy_score: f32,
}

impl<A> Display for ActionWithStaticPolicy<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{action: {}, static_score: {}}}",
            self.action, self.static_policy_score
        )
    }
}

impl<A> Deref for ActionWithStaticPolicy<A> {
    type Target = A;
    fn deref(&self) -> &Self::Target {
        &self.action
    }
}

pub trait MoveProcessor<D: DecisionProcess, E> {
    type Iter: Iterator<Item = E>;
    // This returns an optional outcome (if sampling is merged in the move processor itself, the
    // iterator of generated moves, and the bool representing if the returned outcome is solid
    // and the line can be terminated at this point with the returned outcome.
    fn generate_moves(&self, d: &D, s: &mut D::State) -> (Option<D::Outcome>, Self::Iter, bool);
}

pub trait BlockMoveProcessor<D: DecisionProcess, E> {
    type Iter: Iterator<Item = E>;
    fn generate_moves(
        &self,
        d: &D,
        s: &mut Vec<D::State>,
    ) -> Vec<(Option<D::Outcome>, Self::Iter, bool)>;
}

pub(crate) struct NoProcessing;
impl<D: DecisionProcess> MoveProcessor<D, OnlyAction<D::Action>> for NoProcessing {
    type Iter = <Vec<OnlyAction<D::Action>> as IntoIterator>::IntoIter;

    fn generate_moves(
        &self,
        d: &D,
        s: &mut <D as DecisionProcess>::State,
    ) -> (Option<D::Outcome>, Self::Iter, bool) {
        let outcome_opt = d.is_finished(s);
        let terminal = outcome_opt.is_some();
        (
            outcome_opt,
            d.legal_actions(s)
                .map(|a| OnlyAction { action: a })
                .collect::<Vec<_>>()
                .into_iter(),
            terminal,
        )
    }
}

impl<D: DecisionProcess> BlockMoveProcessor<D, OnlyAction<D::Action>> for NoProcessing {
    type Iter = <Vec<OnlyAction<D::Action>> as IntoIterator>::IntoIter;

    fn generate_moves(
        &self,
        d: &D,
        states: &mut Vec<<D as DecisionProcess>::State>,
    ) -> Vec<(Option<<D as DecisionProcess>::Outcome>, Self::Iter, bool)> {
        states
            .iter()
            .map(|s| {
                let outcome_opt = d.is_finished(s);
                let terminal = outcome_opt.is_some();
                (
                    outcome_opt,
                    d.legal_actions(s)
                        .map(|a| OnlyAction { action: a })
                        .collect::<Vec<_>>()
                        .into_iter(),
                    terminal,
                )
            })
            .collect()
    }
}

pub(crate) struct NoFilteringAndUniformPolicyForPuct;
impl<D: DecisionProcess> BlockMoveProcessor<D, ActionWithStaticPolicy<D::Action>>
    for NoFilteringAndUniformPolicyForPuct
{
    type Iter = <Vec<ActionWithStaticPolicy<D::Action>> as IntoIterator>::IntoIter;

    fn generate_moves(
        &self,
        d: &D,
        states: &mut Vec<<D as DecisionProcess>::State>,
    ) -> Vec<(Option<<D as DecisionProcess>::Outcome>, Self::Iter, bool)> {
        states
            .iter()
            .map(|s| {
                let outcome_opt = d.is_finished(s);
                let terminal = outcome_opt.is_some();
                let factor = 1.0 / d.legal_actions(s).count() as f32;
                (
                    outcome_opt,
                    d.legal_actions(s)
                        .map(|a| ActionWithStaticPolicy {
                            action: a,
                            static_policy_score: factor,
                        })
                        .collect::<Vec<_>>()
                        .into_iter(),
                    terminal,
                )
            })
            .collect()
    }
}
impl<D: DecisionProcess> MoveProcessor<D, ActionWithStaticPolicy<D::Action>>
    for NoFilteringAndUniformPolicyForPuct
{
    type Iter = <Vec<ActionWithStaticPolicy<D::Action>> as IntoIterator>::IntoIter;

    fn generate_moves(
        &self,
        d: &D,
        s: &mut D::State,
    ) -> (Option<<D as DecisionProcess>::Outcome>, Self::Iter, bool) {
        let outcome_opt = d.is_finished(s);
        let terminal = outcome_opt.is_some();
        let factor = 1.0 / d.legal_actions(s).count() as f32;
        (
            outcome_opt,
            d.legal_actions(s)
                .map(|a| ActionWithStaticPolicy {
                    action: a,
                    static_policy_score: factor,
                })
                .collect::<Vec<_>>()
                .into_iter(),
            terminal,
        )
    }
}

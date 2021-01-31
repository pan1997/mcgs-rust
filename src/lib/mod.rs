use crate::lib::decision_process::DecisionProcess;
use crate::lib::mcts::node_store::{ActionWithStaticPolicy, OnlyAction};

mod decision_process;
mod mcts;

trait MoveProcessor<D: DecisionProcess, E> {
    type Iter: Iterator<Item = E>;
    // This returns an optional outcome (if sampling is merged in the move processor itself, the
    // iterator of generated moves, and the bool representing if the returned outcome is solid
    // and the line can be terminated at this point with the returned outcome.
    fn generate_moves(&self, d: &D, s: &mut D::State) -> (Option<D::Outcome>, Self::Iter, bool);
}

trait BlockMoveProcessor<D: DecisionProcess, E> {
    type Iter: Iterator<Item = E>;
    fn generate_moves(
        &self,
        d: &D,
        s: &mut Vec<D::State>,
    ) -> Vec<(Option<D::Outcome>, Self::Iter, bool)>;
}

struct NoProcessing;
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

struct NoFilteringAndUniformPolicyForPuct;
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

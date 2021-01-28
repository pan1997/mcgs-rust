use crate::lib::decision_process::DecisionProcess;
use crate::lib::mcts::node_store::OnlyAction;

mod decision_process;
mod mcts;
mod toy_problems;

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

trait Rollout<D: DecisionProcess> {
    fn sample_outcome(&self, d: &D, s: &mut D::State) -> D::Outcome;
}

struct DefaultRollout;
impl<D: DecisionProcess> Rollout<D> for DefaultRollout
where
    D::Outcome: Default,
{
    fn sample_outcome(&self, _: &D, _: &mut <D as DecisionProcess>::State) -> D::Outcome {
        Default::default()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

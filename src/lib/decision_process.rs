pub trait DecisionProcess {
    type Agent: Copy;
    type Action;
    /// This is the type of data that is used to undo a state transition. This may or may not be
    /// same as [`Action`] type.
    type UndoAction;
    type State;
    type Outcome: Outcome<Self::Agent>;

    fn start_state(&self) -> Self::State;

    type Actions: Iterator<Item = Self::Action>;
    fn legal_actions(&self, s: &Self::State) -> Self::Actions;
    fn agent_to_act(&self, s: &Self::State) -> Self::Agent;

    fn transition(&self, s: &mut Self::State, a: &Self::Action) -> Self::UndoAction;
    fn undo_transition(&self, s: &mut Self::State, u: Self::UndoAction);

    /// The `total` payout (possibly accumulated) when reaching state `s`. Although this needs to
    /// be defined only for states that are terminal, problems with immediate payouts can have an
    /// accumulator associated with the state itself, which can be queried by this `fn`.
    /// Note that the game ends when this returns `Some` outcome.
    fn is_finished(&self, s: &Self::State) -> Option<Self::Outcome>;
}

pub trait Outcome<Agent> {
    type RewardType: Copy;
    fn reward_for_agent(&self, a: Agent) -> Self::RewardType;
}

pub trait Simulator<D: DecisionProcess> {
    fn sample_outcome(&self, d: &D, state: &D::State) -> D::Outcome;
}

/// Defines three categories of outcomes for each agent, win, loss and draw.
pub trait WinnableOutcome<Agent>: Outcome<Agent> {
    fn is_winning_for(&self, a: Agent) -> bool;
}

pub trait ComparableOutcome<Agent>: Outcome<Agent> {
    fn is_better_than(&self, other: &Self, a: Agent) -> bool;
}

impl<T: Copy> Outcome<()> for T {
    type RewardType = T;

    fn reward_for_agent(&self, _: ()) -> Self::RewardType {
        *self
    }
}

pub(crate) struct DefaultSimulator;
impl<D> Simulator<D> for DefaultSimulator
where
    D: DecisionProcess,
    D::Outcome: Default,
{
    fn sample_outcome(&self, _: &D, _: &D::State) -> D::Outcome {
        Default::default()
    }
}

pub(crate) mod graph_dp;
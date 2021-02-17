
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

pub trait SimpleMovingAverage {
    // Updates this value using the simple moving average formula
    fn update_with_moving_average(&mut self, o: &Self, x: u32, y: u32);
    fn reset(&mut self, o: &Self);
}

pub trait Distance: Sized {
    type NormType: PartialOrd;
    fn norm(&self) -> Self::NormType;
    fn sub(&self, other: &Self) -> Self;
    fn clip(&mut self, n: Self::NormType);
    fn scale(&self, w: Self::NormType) -> Self;
}

pub trait Simulator<D: DecisionProcess> {
    fn sample_outcome(&self, d: &D, state: &mut D::State) -> D::Outcome;
}

/// Defines three categories of outcomes for each agent, win, loss and draw.
pub trait WinnableOutcome<Agent>: Outcome<Agent> {
    fn is_winning_for(&self, a: Agent) -> bool;
    fn is_losing_for(&self, a: Agent) -> bool;
}

pub trait ComparableOutcome<Agent>: Outcome<Agent> {
    fn is_better_than(&self, other: &Self, a: Agent) -> bool;
}

impl SimpleMovingAverage for f32 {
    fn update_with_moving_average(&mut self, o: &Self, x: u32, y: u32) {
        *self += (x as f32) * (*o - *self) / (y as f32)
    }

    fn reset(&mut self, o: &Self) {
        *self = *o
    }
}

impl SimpleMovingAverage for f64 {
    fn update_with_moving_average(&mut self, o: &Self, x: u32, y: u32) {
        *self += (x as f64) * (*o - *self) / (y as f64)
    }

    fn reset(&mut self, o: &Self) {
        *self = *o
    }
}

impl SimpleMovingAverage for Vec<f32> {
    fn update_with_moving_average(&mut self, o: &Self, x: u32, y: u32) {
        for (s, other) in self.iter_mut().zip(o.iter()) {
            *s += (x as f32) * (other - *s) / (y as f32)
        }
    }

    fn reset(&mut self, o: &Self) {
        for (x, y) in self.iter_mut().zip(o.iter()) {
            *x = *y;
        }
    }
}

pub struct DefaultSimulator;
impl<D> Simulator<D> for DefaultSimulator
where
    D: DecisionProcess,
    D::Outcome: Default,
{
    fn sample_outcome(&self, _: &D, _: &mut D::State) -> D::Outcome {
        Default::default()
    }
}

use rand::prelude::IteratorRandom;
use rand::prelude::SliceRandom;

pub struct RandomSimulator;
impl<D: DecisionProcess> Simulator<D> for RandomSimulator {
    fn sample_outcome(&self, d: &D, s: &mut D::State) -> D::Outcome {
        let check = d.is_finished(s);
        if check.is_some() {
            check.unwrap()
        } else {
            let mut stack = vec![];
            loop {
                let action = d.legal_actions(s).choose(&mut rand::thread_rng()).unwrap();
                let u = d.transition(s, &action);
                stack.push(u);
                let check = d.is_finished(s);
                if check.is_some() {
                    while let Some(u) = stack.pop() {
                        d.undo_transition(s, u);
                    }
                    return check.unwrap();
                }
            }
        }
    }
}

pub struct OneStepGreedySimulator;
impl<D: DecisionProcess> Simulator<D> for OneStepGreedySimulator
where
    D::Outcome: WinnableOutcome<D::Agent>,
{
    fn sample_outcome(
        &self,
        d: &D,
        s: &mut <D as DecisionProcess>::State,
    ) -> <D as DecisionProcess>::Outcome {
        let check = d.is_finished(s);
        if check.is_some() {
            check.unwrap()
        } else {
            let mut stack = vec![];
            loop {
                let agent = d.agent_to_act(s);
                let actions: Vec<D::Action> = d.legal_actions(s).collect();

                for action in actions.iter() {
                    let u = d.transition(s, action);
                    if let Some(outcome) = d.is_finished(s) {
                        if outcome.is_winning_for(agent) {
                            d.undo_transition(s, u);
                            while let Some(u) = stack.pop() {
                                d.undo_transition(s, u);
                            }
                            return outcome;
                        }
                    }
                    d.undo_transition(s, u);
                }

                let action = actions.choose(&mut rand::thread_rng()).unwrap();
                let u = d.transition(s, &action);
                stack.push(u);
                let check = d.is_finished(s);
                if check.is_some() {
                    while let Some(u) = stack.pop() {
                        d.undo_transition(s, u);
                    }
                    return check.unwrap();
                }
            }
        }
    }
}

impl Distance for f32 {
    type NormType = f32;

    fn norm(&self) -> Self::NormType {
        self.abs()
    }

    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    fn clip(&mut self, n: Self::NormType) {
        if *self > n {
            *self = n
        } else if *self < -n {
            *self = -n
        }
    }

    fn scale(&self, w: Self::NormType) -> Self {
        w * *self
    }
}

impl Distance for f64 {
    type NormType = f64;

    fn norm(&self) -> Self::NormType {
        self.abs()
    }

    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }

    fn clip(&mut self, n: Self::NormType) {
        if *self > n {
            *self = n
        } else if *self < -n {
            *self = -n
        }
    }

    fn scale(&self, w: Self::NormType) -> Self {
        w * *self
    }
}

impl Distance for Vec<f32> {
    type NormType = f32;

    fn norm(&self) -> Self::NormType {
        let mut s = 0.0;
        for x in self.iter() {
            s += x * x;
        }
        s.sqrt()
    }

    fn sub(&self, other: &Self) -> Self {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }

    fn clip(&mut self, n: Self::NormType) {
        self.iter_mut().for_each(|x| x.clip(n))
    }

    fn scale(&self, w: Self::NormType) -> Self {
        self.iter().map(|x| x * w).collect()
    }
}

pub(crate) mod c4;
pub(crate) mod graph_dp;
pub mod hex;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t1() {
        let mut x = 0.0;
        x.update_with_moving_average(&1.0, 1, 1);
        println!("{}", x);
        x.update_with_moving_average(&2.0, 1, 2);
        println!("{}", x);
        x.update_with_moving_average(&3.0, 1, 3);
        println!("{}", x);
    }
}

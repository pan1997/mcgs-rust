use crate::lib::decision_process::SimpleMovingAverage;
use crate::lib::mcgs::search_graph::{OutcomeStore, SelectCountStore};
use std::fmt::{Debug, Display, Formatter};

pub struct Internal<O> {
    pub(crate) expected_sample: O,
    pub(crate) sample_count: u32,
    pub(crate) selection_count: u32,
}

impl<O> Internal<O> {
    pub(crate) fn new(outcome: O) -> Self {
        Internal {
            expected_sample: outcome,
            sample_count: 0,
            selection_count: 0,
        }
    }

    pub(crate) fn is_solved(&self) -> bool {
        self.sample_count == u32::MAX
    }

    pub(crate) fn mark_solved(&mut self) {
        self.sample_count = u32::MAX
    }
}

pub trait HasInternal<O> {
    fn internal(&self) -> &Internal<O>;
    fn internal_mut(&self) -> &mut Internal<O>;
}

impl<O: SimpleMovingAverage + Clone> Internal<O> {
    pub(crate) fn add_sample(&mut self, x: &O, weight: u32) {
        if !self.is_solved() {
            self.sample_count += weight;
            self.expected_sample
                .update_with_moving_average(x, weight, self.sample_count)
        }
    }

    pub(crate) fn fix(&mut self, x: &O) {
        if !self.is_solved() {
            self.expected_sample = x.clone();
            self.mark_solved();
        }
    }
}

impl<T: HasInternal<O>, O: Clone + SimpleMovingAverage> OutcomeStore<O> for T {
    fn expected_outcome(&self) -> O {
        self.internal().expected_sample.clone()
    }

    fn is_solved(&self) -> bool {
        self.internal().is_solved()
    }

    fn add_sample(&self, outcome: &O, weight: u32) {
        self.internal_mut().add_sample(outcome, weight)
    }

    fn sample_count(&self) -> u32 {
        self.internal().sample_count
    }

    fn mark_solved(&self, outcome: &O) {
        self.internal_mut().fix(outcome)
    }
}

impl<O: Display> Display for Internal<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{sel_count: {}, score: {:.2}",
            self.selection_count, self.expected_sample
        )?;
        if self.is_solved() {
            write!(f, "}}")
        } else {
            write!(f, ", sam_count: {}", self.sample_count)
        }
    }
}

impl<O: Debug> Debug for Internal<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{sel_count: {}, score: {:.2?}",
            self.selection_count, self.expected_sample
        )?;
        if self.is_solved() {
            write!(f, ", solid}}")
        } else {
            write!(f, ", sam_count: {}", self.sample_count)
        }
    }
}

use crate::lib::decision_process::SimpleMovingAverage;
use std::fmt::{Debug, Display, Formatter};

pub(crate) struct Internal<O> {
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

impl<O: SimpleMovingAverage> Internal<O> {
    pub(crate) fn add_sample(&mut self, x: &O, weight: u32) {
        if !self.is_solved() {
            self.sample_count += weight;
            self.expected_sample
                .update_with_moving_average(x, weight, self.sample_count)
        }
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

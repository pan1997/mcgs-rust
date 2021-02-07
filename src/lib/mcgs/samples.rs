use num::ToPrimitive;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Samples {
    data: AtomicU64,
}

impl Samples {
    pub(crate) fn new() -> Self {
        Samples {
            data: AtomicU64::new(0),
        }
    }

    pub(crate) fn add_sample(&self, x: f32, count: u32) {
        // TODO: check order
        self.data
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |u| {
                let low: u32 = u as u32;
                if low == u32::MAX {
                    // Do not change if it has been solved
                    Some(u)
                } else {
                    let high: u32 = (u >> 32) as u32;

                    let result_low: u32 = low + count;

                    let result_high: u32 = unsafe {
                        let o: f32 = std::mem::transmute(high);
                        let x_weight = count.to_f32().unwrap();
                        let new_weight = result_low.to_f32().unwrap();
                        // Simple Moving Averages (SAM)
                        let r = o + x_weight * (x - o) / new_weight;
                        std::mem::transmute(r)
                    };

                    Some(((result_high as u64) << 32) + (result_low as u64))
                }
            }).unwrap();
    }

    pub(crate) fn count(&self) -> u32 {
        self.atomic_tuple().0
    }

    pub(crate) fn expected_sample(&self) -> f32 {
        self.atomic_tuple().1
    }

    fn atomic_tuple(&self) -> (u32, f32) {
        let d = self.data.load(Ordering::SeqCst);
        (d as u32, unsafe { std::mem::transmute((d >> 32) as u32) })
    }

    pub(crate) fn is_solved(&self) -> bool {
        self.count() == u32::MAX
    }

    pub(crate) fn mark_solved(&self) {
        self.data
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |u| {
                Some(((u >> 32) << 32) + (u32::MAX as u64))
            }).unwrap();
    }
}

impl Display for Samples {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (w, x) = self.atomic_tuple();
        if w == u32::MAX {
            write!(f, "{{value: {:.2}}}", x)
        } else {
            write!(f, "{{value: {:.2}, count: {}}}", x, w)
        }
    }
}

use crate::lib::decision_process::{DecisionProcess, Simulator};
use crate::lib::mcts::node_store::{TreePolicy, NodeStore};

mod node_store;

struct Search<D, S, T, NS> {
    dp: D,
    simulator: S,
    tree_policy: T,
    store: NS,
}

impl<D, S, T, NS> Search<D, S, T, NS>
where
    D: DecisionProcess,
    S: Simulator<D>,
    NS: NodeStore,
    T: TreePolicy<NS>
{
    pub fn new(dp: D, simulator: S, tree_policy: T, store: NS) -> Self {
        Search {
            dp,
            simulator,
            tree_policy,
            store
        }
    }

    pub fn dp(&self) -> &D {
        &self.dp
    }

    pub fn store(&self) -> &NS {
        &self.store
    }
}

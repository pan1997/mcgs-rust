use crate::lib::decision_process::{DecisionProcess, Outcome, Simulator};
use crate::lib::mcts::node_store::{Edge, Node, NodeStore, TreePolicy};
use std::ops::Deref;

mod node_store;
mod safe_tree;

struct Search<D, S, T, NS> {
    dp: D,
    simulator: S,
    tree_policy: T,
    store: NS,
}

impl<I, D, S, T, NS> Search<D, S, T, NS>
where
    D: DecisionProcess,
    S: Simulator<D>,
    NS: NodeStore,
    T: TreePolicy<NS>,
    NS::Node: Node<<D::Outcome as Outcome<D::Agent>>::RewardType>,
    NS::Edge: Deref<Target=I> + Edge<<D::Outcome as Outcome<D::Agent>>::RewardType>,
    I: Deref<Target=D::Action>
{
    pub fn new(dp: D, simulator: S, tree_policy: T, store: NS) -> Self {
        Search {
            dp,
            simulator,
            tree_policy,
            store,
        }
    }

    pub fn dp(&self) -> &D {
        &self.dp
    }

    pub fn store(&self) -> &NS {
        &self.store
    }

    pub fn select(
        &self,
        n: NS::NodeRef,
        s: &mut D::State,
        stack: &mut Vec<(NS::NodeRef, D::UndoAction)>,
    ) {
        // Using Deref coercion here

        let mut node = n;
        while node.is_finished_expanding() && !node.is_solved() {
            let edge_ref = self
                .tree_policy
                .sample_edge(&self.store, &node, stack.len() as u32)
                .unwrap();
            // TODO: mutate s
            let edge = self.store.get_edge(&node, edge_ref);
            //node.increment_selection_count();
            edge.increment_selection_count();
            node = self.store.get_target_node(edge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::DefaultSimulator;
    use crate::lib::mcts::node_store::RandomTreePolicy;
    use crate::lib::mcts::safe_tree::ThreadSafeNodeStore;
    use crate::lib::mcts::safe_tree::OnlyAction;
    use crate::lib::toy_problems::graph_dp::tests::problem1;

    #[test]
    fn test_basic() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            RandomTreePolicy,
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
        );
    }
}

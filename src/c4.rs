mod lib;
use crate::lib::decision_process::c4::C4;
use crate::lib::decision_process::RandomSimulator;
use crate::lib::decision_process::{DecisionProcess, DefaultSimulator};
use crate::lib::mcts::node_store::{ActionWithStaticPolicy, Node, NodeStore, OnlyAction};
use crate::lib::mcts::safe_tree::ThreadSafeNodeStore;
use crate::lib::mcts::tree_policy::{PuctTreePolicy, UctTreePolicy};
use crate::lib::mcts::Search;
use crate::lib::NoFilteringAndUniformPolicyForPuct;
use crate::lib::NoProcessing;
use std::time::Instant;

fn main() {
    let s = Search::new(
        C4::new(9, 7),
        RandomSimulator,
        PuctTreePolicy::new(2.4),
        ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
        NoFilteringAndUniformPolicyForPuct,
    );

    let node = s.store().new_node();
    let mut state = s.dp().start_state();
    let start_instant = Instant::now();
    s.ensure_valid_starting_node(node.clone(), &mut state);
    for i in 0..2000000 {
        assert_eq!(node.total_selection_count(), i + 1);
        s.once(node.clone(), &mut state);
    }
    s.print_pv(node, Some(start_instant), None, 20);
}

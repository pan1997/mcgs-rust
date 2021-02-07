mod lib;
use crate::lib::decision_process::c4::{Move, C4};
use crate::lib::decision_process::DecisionProcess;
use crate::lib::decision_process::RandomSimulator;
use crate::lib::mcgs::expansion_traits::{BasicExpansion, BasicExpansionWithUniformPrior};
use crate::lib::mcgs::graph_policy::{MostVisitedPolicy, UctPolicy};
use crate::lib::mcgs::safe_tree::SafeTree;
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph, SelectCountStore};
use crate::lib::mcgs::Search;
use crate::lib::mcts::node_store::ActionWithStaticPolicy;
use std::time::Instant;
use text_io::read;

fn main() {
    let s = Search::new(
        C4::new(9, 7),
        SafeTree::<ActionWithStaticPolicy<_>, Vec<f32>>::new(),
        UctPolicy::new(2.4),
        BasicExpansionWithUniformPrior::new(RandomSimulator),
        0.01,
        1,
    );

    let mut state = s.problem().start_state();

    println!("C4 mcts ST");
    loop {
        let token: String = read!();
        match token.as_str() {
            "clear" => state = s.problem().start_state(),
            "exit" | "quit" => std::process::exit(0),
            "drop" => {
                let col: u8 = read!();
                let m = Move(col);
                s.problem().transition(&mut state, &m);
            }
            "board" => println!("{}", state),
            "analyse" => {
                println!("{}", state);
                let time: u128 = read!();
                let node = s.search_graph().create_node(&state);
                let start_instant = Instant::now();
                for _ in 0..2000000 {
                    s.one_iteration(node, &mut state);
                }
                let mut trajectory = vec![];
                s.print_pv(node, Some(start_instant), None, &mut trajectory, 256);
                let best_edge = SearchGraph::<()>::get_edge(
                    s.search_graph(),
                    node,
                    MostVisitedPolicy.select(s.problem(), s.search_graph(), node),
                );
                let best_move: &Move = best_edge;
                let score: f32 = node.expected_outcome();
                println!(
                    "BM {} ms {:.0} total_nodes: {}",
                    best_move,
                    score * 100.0,
                    node.selection_count()
                );
            }
            _ => (),
        }
    }
}

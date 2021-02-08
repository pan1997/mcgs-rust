mod lib;
use crate::lib::decision_process::c4::{Move, C4};
use crate::lib::decision_process::RandomSimulator;
use crate::lib::decision_process::{DecisionProcess, DefaultSimulator};
use crate::lib::mcgs::expansion_traits::{
    BasicExpansion, BasicExpansionWithUniformPrior, BlockExpansionFromBasic,
};
use crate::lib::mcgs::graph_policy::{MostVisitedPolicy, PuctPolicy, UctPolicy};
use crate::lib::mcgs::safe_tree::SafeTree;
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph, SelectCountStore};
use crate::lib::mcgs::Search;
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use std::time::Instant;
use text_io::read;

fn main() {
    /*
    let s = Search::new(
        C4::new(9, 7),
        SafeTree::<ActionWithStaticPolicy<_>, Vec<f32>>::new(),
        PuctPolicy::new(2.4, 20.0),
        BasicExpansionWithUniformPrior::new(RandomSimulator),
        0.01,
        1,
    );*/

    let s = Search::new(
        C4::new(9, 7),
        SafeTree::<OnlyAction<_>, Vec<f32>>::new(),
        UctPolicy::new(2.4),
        BlockExpansionFromBasic::new(BasicExpansion::new(DefaultSimulator)),
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
                let time_limit: u128 = read!();
                let node = s.search_graph().create_node(&state);
                //let elapsed = s.start_block(&node, &mut state, None, Some(time_limit), Some(97));
                //let elapsed = s.start(&node, &mut state, None, Some(time_limit), Some(97));
                let elapsed = s.start_parallel(&node, &state, None, Some(time_limit), Some(97), 8);
                let best_edge = SearchGraph::<()>::get_edge(
                    s.search_graph(),
                    &node,
                    MostVisitedPolicy.select(s.problem(), s.search_graph(), &node),
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

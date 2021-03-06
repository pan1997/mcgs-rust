mod lib;

use crate::lib::decision_process::hex::HexRandomSimulator;
use crate::lib::decision_process::hex::{Hex, Move, ZobHash};
use crate::lib::decision_process::DecisionProcess;
use crate::lib::mcgs::expansion_traits::{BasicExpansion, BlockExpansionFromBasic};
use crate::lib::mcgs::graph::SafeGraph;
use crate::lib::mcgs::graph_policy::{
    PvPolicy, RandomPolicy, SelectionPolicy, UctPolicy, WeightedRandomPolicyWithExpDepth,
};
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph, SelectCountStore};
use crate::lib::mcgs::GraphBasedPrune;
use crate::lib::mcgs::MiniMaxPropagationTask;
use crate::lib::mcgs::Search;
use crate::lib::OnlyAction;
use text_io::read;

fn main() {
    let default_cpu = 12;
    let mut s = Search::new(
        Hex::new(11, 11),
        SafeGraph::<_, OnlyAction<_>, _>::new(0.0),
        WeightedRandomPolicyWithExpDepth::new(RandomPolicy, UctPolicy::new(2.4), 0.05, -1.5),
        BlockExpansionFromBasic::new(BasicExpansion::new(HexRandomSimulator)),
        ZobHash::new(11, 11),
        MiniMaxPropagationTask::new(),
        GraphBasedPrune {
            delta: 0.05,
            clip: 1.0,
            margin: 0,
        },
        default_cpu,
    );

    let mut state = s.problem().start_state();

    println!("Initialised for {} cpu", default_cpu);
    loop {
        let token: String = read!();
        match token.as_str() {
            "clear" => state = s.problem().start_state(),
            "exit" | "quit" => std::process::exit(0),
            "drop" => {
                let (row, col) = (read!(), read!());
                let m = Move(row, col);
                s.problem().transition(&mut state, &m);
            }
            "cpu" => {
                let count = read!();
                s.set_worker_count(count);
                println!("Using {} cpus", count)
            }
            "board" => println!("{}", state),
            "analyse" => {
                println!("{}", state);
                let time_limit: u128 = read!();
                let node = s.get_new_node(&mut state);
                let elapsed =
                    s.start_parallel(&node, &state, None, Some(time_limit), Some(97), false);
                let agent = s.problem().agent_to_act(&state);
                let best_edge = s.search_graph().get_edge(
                    &node,
                    PvPolicy.select(s.problem(), s.search_graph(), &node, agent, 0), //MostVisitedPolicy.select(s.problem(), s.search_graph(), &node),
                );
                let best_move: &Move = best_edge;
                let score = node.expected_outcome();
                println!(
                    "BM {} ms {:.0} total_nodes: {} total_time: {}ms",
                    best_move,
                    score * 100.0,
                    node.selection_count(),
                    elapsed
                );
                s.search_graph().clear(node);
            }
            _ => (),
        }
    }
}

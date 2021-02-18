mod lib;

use crate::lib::decision_process::hex::{Hex, Move, ZobHash};
use crate::lib::decision_process::{DecisionProcess, DefaultSimulator};
use crate::lib::mcgs::expansion_traits::{
    BasicExpansion, BasicExpansionWithUniformPrior, BlockExpansionFromBasic,
};
use crate::lib::mcgs::graph::{NoHash, SafeGraph};
use crate::lib::mcgs::graph_policy::{
    MostVisitedPolicy, PuctPolicy, PvPolicy, RandomPolicy, SelectionPolicy, UctPolicy,
    WeightedRandomPolicyWithExpDepth,
};
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph, SelectCountStore};
use crate::lib::mcgs::tree::SafeTree;
use crate::lib::mcgs::AlwaysExpand;
use crate::lib::mcgs::GraphBasedPrune;
use crate::lib::mcgs::MiniMaxPropagationTask;
use crate::lib::mcgs::Search;
use crate::lib::{ActionWithStaticPolicy, OnlyAction};
use std::sync::atomic::Ordering;
use text_io::read;
use std::time::Instant;
use crate::lib::decision_process::hex::HexRandomSimulator;

fn main() {
    let default_cpu = 12;
    let mut s = Search::new(
        Hex::new(11, 11),
        //SafeTree::<OnlyAction<_>, _>::new(0.0),
        SafeGraph::<_, OnlyAction<_>, _>::new(0.0),
        WeightedRandomPolicyWithExpDepth::new(RandomPolicy, UctPolicy::new(2.4), 0.05, -1.5),
        BlockExpansionFromBasic::new(BasicExpansion::new(HexRandomSimulator)),
        ZobHash::new(11, 11),
        MiniMaxPropagationTask::new(),
        //AlwaysExpand,
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
                let node = s.get_new_node(&mut state); // search_graph().create_node(&state);
                //let t = Instant::now();
                let elapsed =
                    s.start_parallel(&node, &state, None, Some(time_limit), Some(97), false);
                //for i in 0..2000 {
                //    s.one_iteration(&node, &mut state);
                //    if i % 50 == 0 {
                //        println!("{}", i);
                //    }
                //}
                //let elapsed = t.elapsed().as_millis();
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

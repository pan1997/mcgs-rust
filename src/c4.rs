mod lib;
use crate::lib::decision_process::c4::{Move, ZobHash, C4};
use crate::lib::decision_process::DecisionProcess;
use crate::lib::decision_process::RandomSimulator;
use crate::lib::mcgs::expansion_traits::{BasicExpansion, BlockExpansionFromBasic};
use crate::lib::mcgs::graph::SafeGraph;
use crate::lib::mcgs::graph_policy::{
    PuctPolicy, PvPolicy, RandomPolicy, SelectionPolicy, WeightedRandomPolicyWithExpDepth,
};
use crate::lib::mcgs::nnet::c4net::net1;
use crate::lib::mcgs::search_graph::{OutcomeStore, SearchGraph, SelectCountStore};
use crate::lib::mcgs::GraphBasedPrune;
use crate::lib::mcgs::MiniMaxPropagationTask;
use crate::lib::mcgs::Search;
use tch::Device;
use text_io::read;

fn main() {
    /*
    let default_cpu = 8;
    let mut s = Search::new(
        C4::new(9, 7),
        SafeGraph::new(0.0),
        WeightedRandomPolicyWithExpDepth::new(RandomPolicy, UctPolicy::new(2.4), 0.05, -1.5),
        BlockExpansionFromBasic::new(BasicExpansion::new(RandomSimulator)),
        ZobHash::new(9 * 7),
        MiniMaxPropagationTask::new(),
        GraphBasedPrune {
            delta: 0.05,
            clip: 1.0,
            margin: 4,
        },
        default_cpu,
    );*/

    let default_cpu = 1;
    let mut vs = tch::nn::VarStore::new(Device::Cuda(0));
    let root = &vs.root();
    let mut s = Search::new(
        C4::new(9, 7),
        SafeGraph::new(0.0),
        WeightedRandomPolicyWithExpDepth::new(
            RandomPolicy,
            PuctPolicy::new(1280.0, 2.4),
            0.05,
            -1.5,
        ),
        net1(root),
        ZobHash::new(9 * 7),
        MiniMaxPropagationTask::new(),
        GraphBasedPrune {
            delta: 0.05,
            clip: 1.0,
            margin: 4,
        },
        default_cpu,
    );
    let path = "net2";
    vs.load(path).unwrap();

    let mut state = s.problem().start_state();

    println!("Initialised for {} cpu", default_cpu);
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
            "cpu" => {
                let count = read!();
                s.set_worker_count(count);
                println!("Using {} cpus", count)
            }
            "board" => println!("{}", state),
            "analyse" => {
                println!("{}", state);
                let _time_limit: u128 = read!();
                let time_limit = 30000;
                let node = s.get_new_node(&mut state);
                let elapsed =
                    s.start_parallel(&node, &state, None, Some(time_limit), Some(96), true);
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

use crate::lib::decision_process::c4::C4;
use crate::lib::decision_process::{OneStepGreedySimulator, RandomSimulator};
use crate::lib::mcgs::expansion_traits::BasicExpansion;
use crate::lib::mcgs::graph::NoHash;
use crate::lib::mcgs::graph_policy::UctPolicy;
use crate::lib::mcgs::nnet::c4_generate_self_play_game;
use crate::lib::mcgs::nnet::SelfPlayData;
use crate::lib::mcgs::tree::SafeTree;
use crate::lib::mcgs::{AlwaysExpand, Search};
use crate::lib::OnlyAction;
use tch::Tensor;
use text_io::read;

mod lib;

fn generate_games(game_count: u32) -> SelfPlayData {
    let s = Search::new(
        C4::new(9, 7),
        SafeTree::<OnlyAction<_>, _>::new(0.0),
        UctPolicy::new(2.4),
        BasicExpansion::new(OneStepGreedySimulator),
        NoHash,
        (),
        AlwaysExpand,
        1,
    );
    let mut states_tensor = None;
    let mut policy_tensor = None;
    let mut value_tensor = None;
    for _ in 0..game_count {
        let x = c4_generate_self_play_game(&s, 512);
        states_tensor = crate::lib::mcgs::nnet::append(states_tensor, x.states_tensor);
        policy_tensor = crate::lib::mcgs::nnet::append(policy_tensor, x.policy_tensor);
        value_tensor = crate::lib::mcgs::nnet::append(value_tensor, x.value_tensor);
    }
    SelfPlayData {
        states_tensor: states_tensor.unwrap(),
        policy_tensor: policy_tensor.unwrap(),
        value_tensor: value_tensor.unwrap(),
    }
}

fn main() {
    loop {
        let cmd: String = read!();
        match cmd.as_str() {
            "exit" | "quit" => std::process::exit(0),
            "generate" => {
                let count = read!();
                let path: String = read!();
                let x = generate_games(count);
                println!("{:?}", x.states_tensor.size());
                println!("{:?}", x.policy_tensor.size());
                println!("{:?}", x.value_tensor.size());
                Tensor::save(&x.states_tensor, format!("{}.input.zip", path)).unwrap();
                Tensor::save(&x.policy_tensor, format!("{}.policy.zip", path)).unwrap();
                Tensor::save(&x.value_tensor, format!("{}.value.zip", path)).unwrap();
            }
            _ => (),
        }
    }
    let x = generate_games(2);
}

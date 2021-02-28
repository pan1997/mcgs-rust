use crate::lib::decision_process::c4::C4;
use crate::lib::decision_process::{DecisionProcess, RandomSimulator};
use crate::lib::mcgs::expansion_traits::BasicExpansion;
use crate::lib::mcgs::graph::NoHash;
use crate::lib::mcgs::graph_policy::UctPolicy;
use crate::lib::mcgs::nnet::c4net::{c4_generate_self_play_game, net1};
use crate::lib::mcgs::nnet::{process, SelfPlayData};
use crate::lib::mcgs::tree::SafeTree;
use crate::lib::mcgs::{AlwaysExpand, Search};
use crate::lib::OnlyAction;
use tch::nn::OptimizerConfig;
use tch::{Device, Kind, Tensor};
use text_io::read;

mod lib;

fn generate_games(game_count: u32) -> SelfPlayData {
    let s = Search::new(
        C4::new(9, 7),
        SafeTree::<OnlyAction<_>, _>::new(0.0),
        UctPolicy::new(2.4),
        BasicExpansion::new(RandomSimulator),
        NoHash,
        (),
        AlwaysExpand,
        1,
    );
    let mut states_tensors = vec![];
    let mut policy_tensors = vec![];
    let mut value_tensors = vec![];
    for i in 0..game_count {
        if i % 50 == 0 {
            println!("game: {}", i);
        }
        let x = c4_generate_self_play_game(&s, 512);
        states_tensors.push(x.states_tensor);
        policy_tensors.push(x.policy_tensor);
        value_tensors.push(x.value_tensor);
    }
    SelfPlayData {
        states_tensor: Tensor::cat(&states_tensors, 0),
        policy_tensor: Tensor::cat(&policy_tensors, 0),
        value_tensor: Tensor::cat(&value_tensors, 0),
    }
}

fn main() {
    let mut vs = tch::nn::VarStore::new(Device::Cuda(0));
    let root = &vs.root();
    let mut nn = net1(root);
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-3).unwrap();
    loop {
        let cmd: String = read!();
        match cmd.as_str() {
            "exit" | "quit" => std::process::exit(0),
            "generate_blank" => {
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
            "save" => {
                let path: String = read!();
                vs.save(path).unwrap();
            }
            "load" => {
                let path: String = read!();
                vs.load(path).unwrap();
            }
            "test" => {
                let c = C4::new(9, 7);
                let state = c.start_state();
                let input = c
                    .generate_tensor(&state)
                    .to_kind(Kind::Float)
                    .to_device(Device::Cuda(0));
                nn.set_mode(false);
                let (log_policy, v) = nn.forward(&input);
                log_policy.print();
                v.print();
            }
            "update" => {
                let va: i32 = read!();
                let path: String = read!();
                let ic: i32 = read!();
                let states = Tensor::load(format!("{}.input.zip", path)).unwrap();
                let policy = Tensor::load(format!("{}.policy.zip", path)).unwrap();
                let value = Tensor::load(format!("{}.value.zip", path)).unwrap();
                let d = SelfPlayData {
                    states_tensor: states,
                    policy_tensor: policy,
                    value_tensor: value,
                };
                for _ in 0..ic {
                    process(&d, &mut nn, &mut opt, va != 0);
                }
            }
            _ => (),
        }
    }
}

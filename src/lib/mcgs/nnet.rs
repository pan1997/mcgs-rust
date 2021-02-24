pub(crate) mod c4net;
use tch::nn::{ConvConfig, Module, ModuleT, Path, SequentialT};
use tch::{Device, Reduction, Tensor};

pub struct GNet<N1: ModuleT, N2: ModuleT, N3: ModuleT> {
    shared: N1,
    log_policy: N2,
    value: N3,
    training: bool,
}

impl<N1: ModuleT, N2: ModuleT, N3: ModuleT> GNet<N1, N2, N3> {
    fn new(shared: N1, log_policy: N2, value: N3) -> Self {
        GNet {
            shared,
            log_policy,
            value,
            training: false,
        }
    }

    pub(crate) fn set_mode(&mut self, train: bool) {
        self.training = train;
    }

    pub(crate) fn forward(&self, state_representation: &Tensor) -> (Tensor, Tensor) {
        let shared = self.shared.forward_t(state_representation, self.training);
        println!("shared shape: {:?}", shared.size());
        let log_policy = self.log_policy.forward_t(&shared, self.training);
        println!("log_policy shape: {:?}", log_policy.size());
        let value = self.value.forward_t(&shared, self.training);
        println!("value shape: {:?}", value.size());
        (log_policy, value)
    }
}

fn build_sequential_cnn(vs: &Path, kernel_sizes: &[i64], channel_counts: &[i64]) -> SequentialT {
    let mut net = tch::nn::seq_t();
    for index in 0..kernel_sizes.len() {
        let cc = ConvConfig {
            padding: kernel_sizes[index] / 2,
            ..Default::default()
        };
        net = net
            .add(tch::nn::conv2d(
                vs,
                channel_counts[index],
                channel_counts[index + 1],
                kernel_sizes[index],
                cc,
            ))
            .add_fn(|x| x.relu())
    }
    net
}

pub(crate) fn process<T>(
    data: &SelfPlayData,
    net: &mut GNet<impl ModuleT, impl ModuleT, impl ModuleT>,
    opt: &mut tch::nn::Optimizer<T>,
    value: bool,
) {
    net.set_mode(true);

    let input = data.states_tensor.to_device(Device::Cuda(0));
    let policy = data.policy_tensor.to_device(Device::Cuda(0));
    let value_target = data.value_tensor.to_device(Device::Cuda(0));
    let log_policy_target = policy.log().clip(-1000.0, 1000.0);
    let (log_policy_output, value_output) = net.forward(&input);
    let loss = if value {
        value_output.mse_loss(&value_target, Reduction::Mean)
    } else {
        log_policy_output.mse_loss(&log_policy_target, Reduction::Mean)
    };
    opt.backward_step(&loss);
    net.set_mode(false);
}

fn build_ffn(vs: &Path, widths: &[i64]) -> SequentialT {
    let mut net = tch::nn::seq_t();
    for i in 1..widths.len() {
        net = net.add(tch::nn::linear(
            vs,
            widths[i - 1],
            widths[i],
            Default::default(),
        ));
        if i + 1 < widths.len() {
            // No dropout in the last layer
            // also, no relu
            net = net
                .add_fn_t(|x, train| x.dropout(0.8, train))
                .add_fn(|x| x.relu());
        }
    }
    net
}

pub(crate) struct SelfPlayData {
    pub(crate) states_tensor: Tensor,
    pub(crate) policy_tensor: Tensor,
    pub(crate) value_tensor: Tensor,
}

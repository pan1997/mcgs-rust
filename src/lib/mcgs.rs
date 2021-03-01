mod common;
pub mod expansion_traits;
pub mod graph;
pub mod graph_policy;
pub mod nnet;
mod samples;
pub mod search_graph;
pub mod tree;

use graph_policy::*;
use search_graph::*;

use crate::lib::decision_process::{
    ComparableOutcome, DecisionProcess, Distance, Outcome, SimpleMovingAverage, WinnableOutcome,
};
use crate::lib::mcgs::expansion_traits::{BlockExpansionTrait, ExpansionTrait};
use crate::lib::mcgs::graph::Hsh;
use crate::lib::mcgs::SelectionResult::Expand;
use colored::{ColoredString, Colorize};
use num::FromPrimitive;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;
use std::thread::sleep;
use std::time::{Duration, Instant};

pub struct ExpansionResult<O, EI, K> {
    state_key: K,
    outcome: O,
    // To prune, the returned edges iterator needs to be empty now
    edges: EI,
}

pub trait TrajectoryPruner<E, N, V> {
    fn check_for_pruning<'a>(&self, edge: &E, n: &'a N) -> SelectionResult<&'a N, V>;
}

pub struct AlwaysExpand;
impl<E, N, V> TrajectoryPruner<E, N, V> for AlwaysExpand {
    fn check_for_pruning<'a>(&self, _: &E, _: &'a N) -> SelectionResult<&'a N, V> {
        SelectionResult::Expand
    }
}

pub struct GraphBasedPrune<R> {
    pub delta: R,
    pub clip: R,
    pub margin: u32, //maximum_weight: u32
}
impl<E, N, V, R> TrajectoryPruner<E, N, V> for GraphBasedPrune<R>
where
    N: OutcomeStore<V> + SelectCountStore,
    E: OutcomeStore<V> + SelectCountStore,
    V: Distance<NormType = R>,
    R: PartialOrd + FromPrimitive + Copy,
{
    fn check_for_pruning<'a>(&self, edge: &E, n: &'a N) -> SelectionResult<&'a N, V> {
        let e_count = edge.selection_count();
        let n_count = n.selection_count();
        if n_count > e_count + self.margin {
            let e_outcome = edge.expected_outcome();
            let n_outcome = n.expected_outcome();
            let d = n_outcome.sub(&e_outcome).norm();

            if self.delta < d {
                //let e_sample_count = edge.sample_count();
                //let mut o = n_outcome
                //    .scale(FromPrimitive::from_u32(e_sample_count + 1).unwrap())
                //   .sub(&e_outcome.scale(FromPrimitive::from_u32(e_sample_count).unwrap()));
                let mut o = n_outcome
                    .sub(&e_outcome)
                    .scale(FromPrimitive::from_f32(1000.0).unwrap());
                o.clip(self.clip);
                SelectionResult::Propagate(n, o, 1)
            } else {
                SelectionResult::Expand
            }
        } else {
            SelectionResult::Expand
        }
    }
}

pub trait ExtraPropagationTask<G, N, O, A> {
    fn process(&self, graph: &G, node: &N, outcome: &O, agent: A) -> bool;
}
impl<G, N, O, A> ExtraPropagationTask<G, N, O, A> for () {
    fn process(&self, _: &G, _: &N, _: &O, _: A) -> bool {
        false
    }
}

pub struct MiniMaxPropagationTask<P> {
    phantom: PhantomData<P>,
}
impl<P> MiniMaxPropagationTask<P> {
    pub fn new() -> Self {
        MiniMaxPropagationTask {
            phantom: Default::default(),
        }
    }
}
impl<D, H, G: SearchGraph<D, H>, O, A> ExtraPropagationTask<G, G::Node, O, A>
    for MiniMaxPropagationTask<(D, H)>
where
    O: WinnableOutcome<A> + ComparableOutcome<A>,
    G::Node: OutcomeStore<O> + SelectCountStore,
    G::Edge: OutcomeStore<O> + SelectCountStore,
    A: Copy,
{
    fn process(&self, graph: &G, node: &G::Node, outcome: &O, agent: A) -> bool {
        let mut is_solved = true;
        for edge_index in 0..graph.children_count(node) {
            let edge = graph.get_edge(node, edge_index);
            if !edge.is_solved() {
                is_solved = false;
            } else {
                let outcome = edge.expected_outcome();
                if outcome.is_winning_for(agent) {
                    is_solved = true;
                    break;
                }
            }
        }

        if is_solved {
            // should not solidify node as it messes with the transposition detection
            // technically no. it shouldn't mess with it
            node.mark_solved(outcome)
        }
        is_solved
    }
}

// TODO: update this doc
/// The first iteration of this is NOT Thread safe, as it is known that the bottleneck of
/// AlphaZero puct is actually the expansion phase which involves neural network evaluation. A
/// central thread/process is responsible for querying and updating the search tree based on work
/// performed by a set of expansion workers (that may or may not involve neural network
/// computation).
///
/// We also assume that the simulation phase is significantly more expensive than the select and
/// expand, so using a batched expansion running in parallel should be sufficient, (as in
/// https://arxiv.org/abs/1810.11755). This (almost universally true assumption) allows simpler
/// implementation and avoids expensive safeguards to enable safe sharing of the search tree
/// among multiple workers.

pub struct Search<P, S, R, G, X, D, H, Z> {
    problem: P,
    search_graph: G,
    selection_policy: S,
    expand_operation: X,
    state_hasher: H,
    trajectory_pruner: R,
    propagation_task: Z,
    worker_count: u32,
    phantom: PhantomData<D>,
}

pub enum SelectionResult<N, V> {
    Expand,
    Propagate(N, V, u32), // The third arg is the weight
}

impl<P, S, R, G, X, D, H, Z> Search<P, S, R, G, X, D, H, Z>
where
    P: DecisionProcess,
    G: SearchGraph<D, H::K>,
    S: SelectionPolicy<D, P, G, H::K>,
    G::Node: SelectCountStore + OutcomeStore<P::Outcome> + ConcurrentAccess,
    G::Edge: SelectCountStore + OutcomeStore<P::Outcome> + Deref<Target = P::Action>,
    <P::Outcome as Outcome<P::Agent>>::RewardType: Distance,
    P::Outcome: SimpleMovingAverage,
    R: TrajectoryPruner<G::Edge, G::Node, P::Outcome>,
    H: Hsh<P::State>,
    Z: ExtraPropagationTask<G, G::Node, P::Outcome, P::Agent>,
{
    pub fn new(
        problem: P,
        search_graph: G,
        selection_policy: S,
        expand_operation: X,
        state_hasher: H,
        propagation_task: Z,
        trajectory_pruner: R,
        worker_count: u32,
    ) -> Self {
        Search {
            problem,
            search_graph,
            selection_policy,
            expand_operation,
            state_hasher,
            trajectory_pruner,
            propagation_task,
            worker_count,
            phantom: PhantomData,
        }
    }

    pub fn set_worker_count(&mut self, w: u32) {
        self.worker_count = w;
    }

    pub fn problem(&self) -> &P {
        &self.problem
    }

    pub(crate) fn tree_policy(&self) -> &S {
        &self.selection_policy
    }

    pub fn get_new_node(&self, state: &mut P::State) -> G::NodeRef
    where
        X: ExpansionTrait<P, D, H::K>,
    {
        let key = self.state_hasher.key(state);
        let x = self.expand_operation.apply(&self.problem, state, key);
        //assert!(!x.prune);
        let k = self.state_hasher.key(state);
        self.search_graph.create_node(k, x.edges)
    }

    pub fn search_graph(&self) -> &G {
        &self.search_graph
    }

    /// Either creates a new node and returns it for expansion, or returns a
    /// Propagate(outcome, weight). In both cases, the state and node correspond to the state
    /// after reaching the end of trajectory
    fn select<'a>(
        &self,
        root: &'a G::Node,
        state: &mut P::State,
        trajectory: &mut Vec<(&'a G::Node, &'a G::Edge, P::Agent)>,
        undo_stack: &mut Vec<P::UndoAction>,
    ) -> SelectionResult<&'a G::Node, P::Outcome> {
        let mut node = root;
        node.lock();
        while !self.search_graph.is_leaf(node) {
            let agent = self.problem.agent_to_act(state);

            let edge = self.search_graph.get_edge(
                node,
                self.selection_policy.select(
                    &self.problem,
                    &self.search_graph,
                    node,
                    agent,
                    trajectory.len() as u32,
                ),
            );

            //let action: &P::Action = &edge;

            node.increment_selection_count();
            edge.increment_selection_count();

            //let edge_selection_count = edge.selection_count();

            trajectory.push((node, edge, agent));
            undo_stack.push(self.problem.transition(state, &edge));

            if self.search_graph.is_dangling(edge) {
                node.unlock();
                return Expand;
            }

            let next_node = self.search_graph.get_target_node(edge);
            // We need an additional terminal check, as we can mark non leaf nodes as terminal
            if next_node.is_solved() {
                node.unlock();
                return SelectionResult::Propagate(next_node, next_node.expected_outcome(), 1);
            }
            next_node.lock();

            match self.trajectory_pruner.check_for_pruning(edge, next_node) {
                SelectionResult::Propagate(a, b, c) => {
                    next_node.unlock();
                    node.unlock();
                    return SelectionResult::Propagate(a, b, c);
                }
                _ => (),
            };

            // This unlock is done after locking the next node to prevent erroneous transposition
            // detection in the rare case that another thread jumps over this node. This also is
            // safe and doesn't cause a deadlock with back propagation, as propagation unlocks the
            // node before locking another.
            node.unlock();

            node = next_node;
        }
        node.unlock();
        SelectionResult::Propagate(node, node.expected_outcome(), 1)
    }

    fn expand<'a>(
        &self,
        node: &'a G::Node,
        edge: &'a G::Edge,
        state: &mut P::State,
    ) -> (&'a G::Node, P::Outcome, u32)
    where
        X: ExpansionTrait<P, D, H::K>,
    {
        let expansion_result =
            self.expand_operation
                .apply(&self.problem, state, self.state_hasher.key(state));
        node.lock();
        // no problem with locking
        let new_node =
            self.search_graph
                .add_child(expansion_result.state_key, edge, expansion_result.edges);
        new_node.lock();
        node.unlock();
        if self.search_graph.is_leaf(new_node) {
            new_node.mark_solved(&expansion_result.outcome);
        } else {
            new_node.add_sample(&expansion_result.outcome, 1);
        }
        new_node.increment_selection_count();
        new_node.unlock();
        (new_node, expansion_result.outcome, 1)
    }

    fn propagate(
        &self,
        trajectory: &mut Vec<(&G::Node, &G::Edge, P::Agent)>,
        node: &G::Node, // This is needed later for alpha beta
        mut outcome: P::Outcome,
        mut weight: u32,
    ) where
        P::Outcome: Debug,
    {
        let mut last_node = node;
        while let Some((node, edge, agent)) = trajectory.pop() {
            node.lock();
            let n_count = node.selection_count();
            let w = if n_count > 409600 {
                16 * weight
            } else if n_count > 65536 {
                8 * weight
            } else if n_count > 16384 {
                4 * weight
            } else if n_count > 1024 {
                2 * weight
            } else {
                weight
            };
            if last_node.is_solved() {
                edge.mark_solved(&last_node.expected_outcome());
                //println!("marked an edge as solved with: {:?}",edge.expected_outcome());
            } else {
                edge.add_sample(&outcome, w);
            }
            node.add_sample(&outcome, w);
            let marked = self
                .propagation_task
                .process(&self.search_graph, node, &outcome, agent);
            /*if marked {
                println!("marked a node as solved with: {:?}, {:?}", outcome, node.expected_outcome
                ());
            }*/
            node.unlock();
            last_node = node;
        }
    }

    pub fn one_iteration(&self, root: &G::Node, state: &mut P::State)
    where
        X: ExpansionTrait<P, D, H::K>,
        P::Outcome: Debug,
    {
        let trajectory = &mut vec![];
        let undo_stack = &mut vec![];
        let s = self.select(root, state, trajectory, undo_stack);

        let (node, outcome, weight) = match s {
            SelectionResult::Expand => {
                let (n, e, _) = trajectory.last().unwrap();
                self.expand(*n, *e, state)
            }
            SelectionResult::Propagate(node, outcome, weight) => (node, outcome, weight),
        };

        while let Some(u) = undo_stack.pop() {
            self.problem.undo_transition(state, u);
        }

        self.propagate(trajectory, node, outcome, weight);
    }

    fn one_block(&self, root: &G::Node, state: &mut P::State, size: usize)
    where
        X: BlockExpansionTrait<P, D, H::K>,
        P::Outcome: Debug,
    {
        let trajectories = &mut vec![];
        let undo_stack = &mut vec![];
        let mut short_count = 0;
        let mut batch = self.expand_operation.new_batch();
        while trajectories.len() < size && short_count < 24 * size {
            let mut trajectory = vec![];
            let s = self.select(root, state, &mut trajectory, undo_stack);
            match s {
                SelectionResult::Propagate(node, outcome, weight) => {
                    self.propagate(&mut trajectory, node, outcome, weight);
                    short_count += 1;
                }
                SelectionResult::Expand => trajectories.push((
                    trajectory,
                    self.expand_operation.accept(
                        &self.problem,
                        state,
                        self.state_hasher.key(state),
                        &mut batch,
                    ),
                )),
            }
            while let Some(u) = undo_stack.pop() {
                self.problem.undo_transition(state, u);
            }
        }
        let expansion_results = self.expand_operation.process_accepted(batch);

        let mut inverse_map = vec![0; expansion_results.len()];
        // TODO: fix collisions
        for (i, (_, index)) in trajectories.iter().enumerate() {
            inverse_map[*index] = i;
        }

        for (expansion_result, trajectory_index) in expansion_results.into_iter().zip(inverse_map) {
            let (trajectory, _) = &mut trajectories[trajectory_index];

            let (_, e, _) = trajectory.last().unwrap();
            let node =
                self.search_graph
                    .add_child(expansion_result.state_key, e, expansion_result.edges);
            self.propagate(trajectory, node, expansion_result.outcome, 1);
        }
    }

    fn end_search(
        &self,
        start_time: Instant,
        root: &G::Node,
        node_limit: Option<u32>,
        time_limit: Option<u128>,
        confidence: Option<u32>,
        minimum_count: u32,
    ) -> bool {
        if time_limit.is_some() {
            let elapsed = start_time.elapsed().as_millis();
            if elapsed > time_limit.unwrap() {
                return true;
            }
        }
        if node_limit.is_some() {
            let node_count = root.selection_count();
            if node_count > node_limit.unwrap() {
                return true;
            }
        }
        if confidence.is_some() {
            let node_count = root.selection_count();
            if node_count > minimum_count {
                let limit = ((node_count as u64 * confidence.unwrap() as u64) / 100) as u32;
                let best_edge_count = self
                    .search_graph
                    .get_edge(
                        root,
                        MostVisitedPolicy.select(&self.problem, &self.search_graph, root),
                    )
                    .selection_count();
                if best_edge_count > limit {
                    return true;
                }
            }
        }
        if root.is_solved() {
            return true;
        }
        false
    }

    pub fn start_parallel(
        &self,
        root: &G::Node,
        state: &P::State,
        node_limit: Option<u32>,
        time_limit: Option<u128>,
        confidence: Option<u32>, // out of 100
        block: bool,
    ) -> u128
    where
        P::State: Clone + Send,
        X: ExpansionTrait<P, D, H::K> + BlockExpansionTrait<P, D, H::K>,
        P::Action: Display,
        P::Agent: Display,
        P::Outcome: WinnableOutcome<P::Agent> + Display + Debug,
        <P::Outcome as Outcome<P::Agent>>::RewardType: Display,
        <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType: Sync,
        G::Node: Sync,
        P: Sync,
        G: Sync,
        S: Sync,
        X: Sync,
        D: Sync,
        H: Sync,
        Z: Sync,
        R: Sync,
    {
        let start_time = Instant::now();
        let start_count = root.selection_count();
        crossbeam::scope(|scope| {
            for _ in 0..self.worker_count {
                let mut local_state = state.clone();
                scope.spawn(move |_| {
                    while !self.end_search(start_time, root, node_limit, time_limit, confidence, 32)
                    {
                        if !block || root.selection_count() < 256 {
                            for _ in 0..64 {
                                self.one_iteration(root, &mut local_state);
                            }
                        } else {
                            self.one_block(root, &mut local_state, 128);
                        }
                    }
                });
            }
            let mut pv_state = state.clone();
            while !self.end_search(start_time, root, node_limit, time_limit, confidence, 32) {
                sleep(Duration::from_millis(500));
                let mut v = vec![];
                self.print_pv(
                    root,
                    &mut pv_state,
                    Some(start_time),
                    Some(start_count),
                    &mut v,
                    1024,
                    true,
                );
            }
            self.mpv(root, &mut pv_state, Some(start_time), Some(start_count));
        })
        .unwrap();
        start_time.elapsed().as_millis()
    }

    pub fn mpv<'a>(
        &self,
        mut root: &'a G::Node,
        state: &mut P::State,
        start_time: Option<Instant>,
        start_count: Option<u32>,
    ) where
        P::Action: Display,
        P::Agent: Display,
        P::Outcome: WinnableOutcome<P::Agent> + Display,
        <P::Outcome as Outcome<P::Agent>>::RewardType: Display,
    {
        let agent = self.problem.agent_to_act(state);
        for edge_index in 0..self.search_graph.children_count(root) {
            let edge = self.search_graph.get_edge(root, edge_index);
            let node = self.search_graph.get_target_node(edge);
            print!("====>");
            self.print_node(root, edge, agent, edge);
            let trajectory = &mut vec![];
            let u = self.problem.transition(state, &edge);
            self.print_pv(node, state, start_time, start_count, trajectory, 0, false);
            self.problem.undo_transition(state, u);
        }
    }

    fn print_node(&self, node: &G::Node, edge: &G::Edge, agent: P::Agent, action: &P::Action)
    where
        P::Action: Display,
        P::Agent: Display,
        P::Outcome: WinnableOutcome<P::Agent> + Display,
        <P::Outcome as Outcome<P::Agent>>::RewardType: Display,
    {
        let es = edge.selection_count();
        let rs = node.selection_count();
        let confidence = (if es == u32::MAX {
            100
        } else {
            es as u64 * 100 / rs as u64
        }) as u32;
        let entry = if edge.is_solved() {
            format!(
                "[{}, {:.2}]-> ",
                action,
                edge.expected_outcome() //.reward_for_agent(agent)
            )
        } else {
            format!(
                "{{{}, {}k, {:.2}}}-> ",
                action,
                es / 1000,
                edge.expected_outcome().reward_for_agent(agent)
            )
        };
        print!("{}", color_from_confidence(&entry, confidence));
    }

    pub(crate) fn print_pv<'a>(
        &self,
        mut root: &'a G::Node,
        state: &mut P::State,
        start_time: Option<Instant>,
        start_count: Option<u32>,
        trajectory: &mut Vec<(&'a G::Node, &'a G::Edge)>,
        filter: u32,
        detail: bool,
    ) where
        P::Action: Display,
        P::Agent: Display,
        P::Outcome: WinnableOutcome<P::Agent> + Display,
        <P::Outcome as Outcome<P::Agent>>::RewardType: Display,
    {
        // No locks used here as this fn only reads
        let selection_count = root.selection_count() - start_count.or(Some(0)).unwrap();

        let mut sl = 0;
        let mut stack = vec![];
        while !self.search_graph.is_leaf(root) {
            let agent = self.problem.agent_to_act(state);
            let best_edge_index =
                PvPolicy.select(&self.problem, &self.search_graph, root, agent, 0);
            let edge = self.search_graph.get_edge(root, best_edge_index);
            if self.search_graph.is_dangling(edge) {
                break;
            }
            let action: &P::Action = &edge;
            stack.push(self.problem.transition(state, action));
            if root.selection_count() > filter || root.is_solved() {
                sl += 1;
                self.print_node(root, edge, agent, action);
            } else {
                print!("{{?}}-> ");
            }
            trajectory.push((root, edge));
            root = self.search_graph.get_target_node(edge);
        }
        println!();
        if detail {
            print!("{}kN ", selection_count / 1000);
            if start_time.is_some() {
                let elapsed = start_time.unwrap().elapsed().as_millis();
                print!(
                    "in {}ms @{}kNps ",
                    elapsed,
                    selection_count as u128 / (elapsed + 1)
                );
            }
            println!(" depth {}/{}", sl, trajectory.len());
        }
        while let Some(u) = stack.pop() {
            self.problem.undo_transition(state, u);
        }
    }
}

impl<K: Clone, O: Clone, EI: Clone> Clone for ExpansionResult<K, O, EI> {
    fn clone(&self) -> Self {
        ExpansionResult {
            state_key: self.state_key.clone(),
            outcome: self.outcome.clone(),
            edges: self.edges.clone(),
        }
    }
}

fn color_from_confidence(s: &str, c: u32) -> ColoredString {
    if c < 25 {
        s.strikethrough()
    } else if c < 50 {
        s.red()
    } else if c < 65 {
        s.yellow()
    } else if c < 80 {
        s.green()
    } else if c < 90 {
        s.cyan()
    } else if c < 95 {
        s.blue()
    } else {
        s.magenta()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::c4::{Move, ZobHash, C4};
    use crate::lib::decision_process::graph_dp::tests::{problem1, problem2, problem3, DSim};
    use crate::lib::decision_process::graph_dp::GHash;
    use crate::lib::decision_process::hex::{Hex, HexRandomSimulator};
    use crate::lib::decision_process::{DefaultSimulator, OneStepGreedySimulator};
    use crate::lib::mcgs::expansion_traits::{
        BasicExpansion, BasicExpansionWithUniformPrior, BlockExpansionFromBasic,
    };
    use crate::lib::mcgs::graph::tests::print_graph;
    use crate::lib::mcgs::graph::{NoHash, SafeGraph};
    use crate::lib::mcgs::tree::tests::print_tree;
    use crate::lib::mcgs::tree::SafeTree;
    use crate::lib::{ActionWithStaticPolicy, OnlyAction};

    #[test]
    fn random() {
        let s = Search::new(
            problem1(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            RandomPolicy,
            BasicExpansion::new(DSim),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..10 {
            s.one_iteration(&n, state);
            print_tree(&s.search_graph, &n, 0, true);
        }
    }

    #[test]
    fn uct() {
        let s = Search::new(
            problem1(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            UctPolicy::new(2.4),
            BasicExpansion::new(DSim),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..100 {
            s.one_iteration(&n, state);
            print_tree(&s.search_graph, &n, 0, true);
        }
    }

    #[test]
    fn uct_block() {
        let s = Search::new(
            problem1(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            UctPolicy::new(2.4),
            BlockExpansionFromBasic::new(BasicExpansion::new(DSim)),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..20 {
            s.one_block(&n, state, 5);
            print_tree(&s.search_graph, &n, 0, true);
        }
    }

    #[test]
    fn puct_2p() {
        let s = Search::new(
            problem2(),
            SafeTree::<ActionWithStaticPolicy<_>, _>::new(vec![0.0, 0.0]),
            PuctPolicy::new(2.0, 2.4),
            BasicExpansionWithUniformPrior::new(DSim),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..100 {
            s.one_iteration(&n, state);
            print_tree(&s.search_graph, &n, 0, true);
        }
    }

    #[test]
    fn puct() {
        let s = Search::new(
            problem1(),
            SafeTree::<ActionWithStaticPolicy<_>, _>::new(vec![0.0, 0.0]),
            PuctPolicy::new(2000.0, 2.4),
            BasicExpansionWithUniformPrior::new(DSim),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..100 {
            s.one_iteration(&n, state);
            print_tree(&s.search_graph, &n, 0, true);
        }
    }

    #[test]
    fn c4_test() {
        let s = Search::new(
            C4::new(9, 7),
            SafeTree::<OnlyAction<_>, _>::new(0.0),
            UctPolicy::new(2.4),
            BasicExpansion::new(DefaultSimulator),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );
        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..1000 {
            s.one_iteration(&n, state);
        }
        print_tree(&s.search_graph, &n, 0, false);
    }

    #[test]
    fn g_test_3() {
        let s = Search::new(
            problem3(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            UctPolicy::new(2.4),
            BasicExpansion::new(DSim),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..400 {
            s.one_iteration(&n, state);
        }
        print_tree(&s.search_graph, &n, 0, true);
        s.one_iteration(&n, state);
        println!()
    }

    #[test]
    fn g_test() {
        let s = Search::new(
            problem3(),
            SafeGraph::<u32, OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            UctPolicy::new(2.4),
            BasicExpansion::new(DSim),
            GHash,
            (),
            //AlwaysExpand,
            GraphBasedPrune {
                delta: 0.01,
                clip: 1.0,
                margin: 2,
            },
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_graph(&s.search_graph, &n, 0, true);
        for _ in 0..400 {
            s.one_iteration(&n, state);
        }
        print_graph(&s.search_graph, &n, 0, true);
        println!()
    }
    #[test]
    fn mmax_test() {
        let s = Search::new(
            problem3(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            UctPolicy::new(2.4),
            BasicExpansion::new(DSim),
            NoHash,
            MiniMaxPropagationTask::new(),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        let n = s.get_new_node(state);
        print_tree(&s.search_graph, &n, 0, true);
        for _ in 0..40 {
            s.one_iteration(&n, state);
        }
        print_tree(&s.search_graph, &n, 0, true);
        s.one_iteration(&n, state);
        println!()
    }

    #[test]
    fn uct_hex() {
        let s = Search::new(
            Hex::new(5, 5),
            SafeTree::new(0.0),
            UctPolicy::new(2.4),
            BasicExpansion::new(HexRandomSimulator),
            NoHash,
            (),
            AlwaysExpand,
            1,
        );

        let state = &mut s.problem.start_state();
        println!("{}", state);
        let n = s.get_new_node(state);
        for _ in 0..2500 {
            s.one_iteration(&n, state);
        }
        println!("{}", state);
        s.print_pv(&n, state, None, None, &mut vec![], 1, true);
    }

    #[test]
    fn c4t23() {
        let mut s = Search::new(
            C4::new(9, 7),
            SafeGraph::<_, OnlyAction<_>, _>::new(0.0),
            WeightedRandomPolicyWithExpDepth::new(RandomPolicy, UctPolicy::new(2.4), 0.05, -1.5),
            BlockExpansionFromBasic::new(BasicExpansion::new(OneStepGreedySimulator)),
            ZobHash::new(63),
            MiniMaxPropagationTask::new(),
            GraphBasedPrune {
                delta: 0.05,
                clip: 1.0,
                margin: 4,
            },
            1,
        );
        let mut state = s.problem().start_state();
        let moves = [4, 3, 4, 3, 4];
        for m in moves.iter() {
            s.problem.transition(&mut state, &Move(*m as u8));
        }
        println!("{}", state);
        let node = s.get_new_node(&mut state);
        s.start_parallel(&node, &state, Some(2048), None, None, false);
    }
}

mod common;
pub(crate) mod expansion_traits;
pub(crate) mod graph;
pub(crate) mod graph_policy;
mod samples;
pub(crate) mod search_graph;
pub(crate) mod tree;

use graph_policy::*;
use search_graph::*;

use crate::lib::decision_process::{DecisionProcess, Distance, Outcome, SimpleMovingAverage};
use crate::lib::mcgs::expansion_traits::{BlockExpansionTrait, ExpansionTrait};
use crate::lib::mcgs::graph::Hsh;
use crate::lib::mcgs::SelectionResult::Expand;
use std::cmp::min;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Deref;
use std::process::exit;
use std::thread::sleep;
use std::time::{Duration, Instant};

pub struct ExpansionResult<O, EI, K> {
    state_key: K,
    outcome: O,
    // To prune, the returned edges iterator needs to be empty now
    edges: EI,
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

pub struct Search<P, S, R, G, X, D, H> {
    problem: P,
    search_graph: G,
    selection_policy: S,
    expand_operation: X,
    state_hasher: H,
    q_shorting_bound: R,
    maximum_trajectory_weight: u32,
    worker_count: u32,
    phantom: PhantomData<D>,
}

enum SelectionResult<N, V> {
    Expand,
    Propagate(N, V, u32), // The third arg is the weight
}

impl<P, S, G, X, D, H>
    Search<P, S, <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType, G, X, D, H>
where
    P: DecisionProcess,
    G: SearchGraph<D, H::K>,
    S: SelectionPolicy<D, P, G, H::K>,
    G::Node: SelectCountStore + OutcomeStore<P::Outcome> + ConcurrentAccess,
    G::Edge: SelectCountStore + OutcomeStore<P::Outcome> + Deref<Target = P::Action>,
    <P::Outcome as Outcome<P::Agent>>::RewardType: Distance,
    P::Outcome: SimpleMovingAverage,
    H: Hsh<P::State>,
{
    pub(crate) fn new(
        problem: P,
        search_graph: G,
        selection_policy: S,
        expand_operation: X,
        state_hasher: H,
        q_shorting_bound: <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType,
        maximum_trajectory_weight: u32,
        worker_count: u32,
    ) -> Self {
        Search {
            problem,
            search_graph,
            selection_policy,
            expand_operation,
            state_hasher,
            q_shorting_bound,
            maximum_trajectory_weight,
            worker_count,
            phantom: PhantomData,
        }
    }

    pub(crate) fn set_worker_count(&mut self, w: u32) {
        self.worker_count = w;
    }

    pub(crate) fn problem(&self) -> &P {
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

    pub(crate) fn search_graph(&self) -> &G {
        &self.search_graph
    }

    /// Either creates a new node and returns it for expansion, or returns a
    /// Propagate(outcome, weight). In both cases, the state and node correspond to the state
    /// after reaching the end of trajectory
    fn select<'a>(
        &self,
        root: &'a G::Node,
        state: &mut P::State,
        trajectory: &mut Vec<(&'a G::Node, &'a G::Edge)>,
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

            let edge_selection_count = edge.selection_count();

            trajectory.push((node, edge));
            undo_stack.push(self.problem.transition(state, &edge));

            if self.search_graph.is_dangling(edge) {
                node.unlock();
                return Expand;
            }

            let next_node = self.search_graph.get_target_node(edge);
            next_node.lock();

            // This unlock is done after locking the next node to prevent erroneous transposition
            // detection in the rare case that another thread jumps over this node. This also is
            // safe and doesn't cause a deadlock with back propagation, as propagation unlocks the
            // node before locking another.
            node.unlock();

            // Check if this is a transposition (next_node has extra selections)
            if next_node.selection_count() > edge_selection_count {
                // Next node always has some samples, as we have moved away from dangling nodes
                // to dangling edges.
                let delta = next_node
                    .expected_outcome()
                    .reward_for_agent(agent)
                    .distance(&edge.expected_outcome().reward_for_agent(agent));
                if delta > self.q_shorting_bound {
                    println!("Shorting");
                    let node_selection_count = next_node.selection_count();
                    let expected_outcome = next_node.expected_outcome();
                    next_node.unlock();
                    return SelectionResult::Propagate(
                        next_node,
                        expected_outcome,
                        min(
                            self.maximum_trajectory_weight,
                            node_selection_count - edge_selection_count,
                        ),
                    );
                }
            }

            // We need an additional terminal check, as we can mark non leaf nodes as terminal
            if next_node.is_solved() {
                next_node.unlock();
                return SelectionResult::Propagate(next_node, next_node.expected_outcome(), 1);
            }

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
        node.unlock();
        // TODO: this might create races
        new_node.lock();
        new_node.add_sample(&expansion_result.outcome, 1);
        if self.search_graph.is_leaf(new_node) {
            new_node.mark_solved()
        }
        new_node.increment_selection_count();
        new_node.unlock();
        (new_node, expansion_result.outcome, 1)
    }

    fn propagate(
        &self,
        trajectory: &mut Vec<(&G::Node, &G::Edge)>,
        node: &G::Node, // This is needed later for alpha beta
        mut outcome: P::Outcome,
        mut weight: u32,
    ) {
        while let Some((node, edge)) = trajectory.pop() {
            node.lock();
            let nc = node.selection_count();
            let nd = self.search_graph.children_count(node);
            let w = if nc > nd * 16 {
                weight * 4
            } else if nc > nd * 4 {
                weight * 2
            } else {
                weight
            };
            edge.add_sample(&outcome, w);
            node.add_sample(&outcome, w);
            node.unlock();
        }
    }

    pub(crate) fn one_iteration(&self, root: &G::Node, state: &mut P::State)
    where
        X: ExpansionTrait<P, D, H::K>,
    {
        let trajectory = &mut vec![];
        let undo_stack = &mut vec![];
        let s = self.select(root, state, trajectory, undo_stack);

        let (node, outcome, weight) = match s {
            SelectionResult::Expand => {
                let (n, e) = trajectory.last().unwrap();
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
    {
        let trajectories = &mut vec![];
        let undo_stack = &mut vec![];
        let mut short_count = 0;
        let mut batch = self.expand_operation.new_batch();
        while trajectories.len() < size && short_count < 2 * size {
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

            let (_, e) = trajectory.last().unwrap();
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

    pub(crate) fn start_block(
        &self,
        root: &G::Node,
        state: &mut P::State,
        node_limit: Option<u32>,
        time_limit: Option<u128>,
        confidence: Option<u32>, // out of 100
    ) -> u128
    where
        X: BlockExpansionTrait<P, D, H::K> + ExpansionTrait<P, D, H::K>,
        P::Action: Display,
        P::Outcome: Display,
    {
        let start_time = Instant::now();
        let start_count = root.selection_count();
        let mut next_pv = 1;
        let mut worker = || {
            while !self.end_search(start_time, root, node_limit, time_limit, confidence, 32) {
                if root.selection_count() < 256 {
                    for _ in 0..256 {
                        self.one_iteration(root, state);
                    }
                } else {
                    self.one_block(root, state, 32);
                }
                let index = start_time.elapsed().as_millis() / 500;
                if index >= next_pv {
                    next_pv += 1;
                    let mut v = vec![];
                    self.print_pv(root, Some(start_time), Some(start_count), &mut v, 256);
                }
            }
        };
        worker();
        start_time.elapsed().as_millis()
    }

    pub(crate) fn start_parallel(
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
        P::Outcome: Display,
        <<P::Outcome as Outcome<P::Agent>>::RewardType as Distance>::NormType: Sync,
        G::Node: Sync,
        P: Sync,
        G: Sync,
        S: Sync,
        X: Sync,
        D: Sync,
        H: Sync,
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
                            for _ in 0..256 {
                                self.one_iteration(root, &mut local_state);
                            }
                        } else {
                            self.one_block(root, &mut local_state, 64);
                        }
                    }
                });
            }
            while !self.end_search(start_time, root, node_limit, time_limit, confidence, 32) {
                sleep(Duration::from_millis(500));
                let mut v = vec![];
                self.print_pv(root, Some(start_time), Some(start_count), &mut v, 1024);
            }
        })
        .unwrap();
        start_time.elapsed().as_millis()
    }

    pub(crate) fn print_pv<'a>(
        &self,
        mut root: &'a G::Node,
        start_time: Option<Instant>,
        start_count: Option<u32>,
        trajectory: &mut Vec<(&'a G::Node, &'a G::Edge)>,
        filter: u32,
    ) where
        P::Action: Display,
        P::Outcome: Display,
    {
        // No locks used here as this fn only reads
        let selection_count = root.selection_count() - start_count.or(Some(0)).unwrap();
        print!("{}kN ", selection_count / 1000);
        if start_time.is_some() {
            let elapsed = start_time.unwrap().elapsed().as_millis();
            print!(
                "in {}ms @{}kNps ",
                elapsed,
                selection_count as u128 / (elapsed + 1)
            );
        }
        let mut sl = 0;
        while !self.search_graph.is_leaf(root) {
            let best_edge_index = MostVisitedPolicy.select(&self.problem, &self.search_graph, root);
            let edge = self.search_graph.get_edge(root, best_edge_index);
            if self.search_graph.is_dangling(edge) {
                break;
            }
            let action: &P::Action = &edge;
            if root.selection_count() > filter {
                sl += 1;
                print!(
                    "{{{}, {}k({})%, {:.2}",
                    action,
                    edge.selection_count() / 1000,
                    edge.selection_count() * 100 / (root.selection_count() + 1), // avoid nast divide by zero
                    root.expected_outcome()
                );
                if root.is_solved() {
                    print!("S}}-> ");
                } else {
                    print!("}}-> ");
                }
            } else {
                print!("{{?}}-> ");
            }
            trajectory.push((root, edge));
            root = self.search_graph.get_target_node(edge);
        }
        println!("| depth {}/{}", sl, trajectory.len());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::c4::C4;
    use crate::lib::decision_process::graph_dp::tests::{problem1, problem2, problem3, DSim};
    use crate::lib::decision_process::DefaultSimulator;
    use crate::lib::mcgs::expansion_traits::{
        BasicExpansion, BasicExpansionWithUniformPrior, BlockExpansionFromBasic,
    };
    use crate::lib::mcgs::graph::{NoHash, SafeGraph};
    use crate::lib::mcgs::tree::tests::print_tree;
    use crate::lib::mcgs::tree::SafeTree;
    use crate::lib::{ActionWithStaticPolicy, OnlyAction};
    use crate::lib::decision_process::graph_dp::GHash;
    use crate::lib::mcgs::graph::tests::print_graph;

    #[test]
    fn random() {
        let s = Search::new(
            problem1(),
            SafeTree::<OnlyAction<_>, _>::new(vec![0.0, 0.0]),
            RandomPolicy,
            BasicExpansion::new(DSim),
            NoHash,
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
            0.01,
            1,
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
}

use crate::lib::decision_process::{DecisionProcess, Outcome, Simulator};
use crate::lib::mcts::node_store::{Edge, Node, NodeStore, TreePolicy};
use crate::lib::{BlockMoveProcessor, MoveProcessor};
use std::fmt::Display;
use std::ops::Deref;

pub(crate) mod node_store;
pub(crate) mod safe_tree;
pub(crate) mod tree_policy;

pub(crate) struct Search<D, S, T, NS, P> {
    dp: D,
    simulator: S,
    tree_policy: T,
    store: NS,
    preprocessor: P,
}

impl<I, D, S, T, NS, P> Search<D, S, T, NS, P>
where
    D: DecisionProcess,
    S: Simulator<D>,
    NS: NodeStore,
    T: TreePolicy<NS>,
    NS::Node: Node<<D::Outcome as Outcome<D::Agent>>::RewardType>,
    NS::Edge: Deref<Target = I> + Edge<<D::Outcome as Outcome<D::Agent>>::RewardType>,
    I: Deref<Target = D::Action> + Into<NS::Edge>,
{
    pub fn new(dp: D, simulator: S, tree_policy: T, store: NS, preprocessor: P) -> Self {
        Search {
            dp,
            simulator,
            tree_policy,
            store,
            preprocessor,
        }
    }

    pub fn dp(&self) -> &D {
        &self.dp
    }

    pub fn store(&self) -> &NS {
        &self.store
    }

    fn select(
        &self,
        n: NS::NodeRef,
        s: &mut D::State,
        stack: &mut Vec<(NS::NodeRef, D::UndoAction)>,
    ) -> NS::NodeRef {
        // Using Deref coercion here

        let mut node = n;
        while node.is_finished_expanding() && !node.is_solved() {
            let edge_ref = self
                .tree_policy
                .sample_edge(&self.store, &node, stack.len() as u32)
                .unwrap();

            let edge = self.store.get_edge_mut(&node, edge_ref);
            let undo_action = self.dp.transition(s, &edge);

            node.increment_selection_count();
            edge.increment_selection_count();
            stack.push((node, undo_action));
            node = self.store.get_or_create_target_node(edge);
        }
        node
    }

    fn backtrack(
        &self,
        n: NS::NodeRef,
        s: &mut D::State,
        stack: &mut Vec<(NS::NodeRef, D::UndoAction)>,
        o: D::Outcome,
        w: u32,
    ) {
        let mut prev = n;
        while let Some((n, u)) = stack.pop() {
            self.dp.undo_transition(s, u);
            // previous node (child node gets a new sample that is form the perspective the
            // player at the state corresponding to the parent node
            prev.add_sample(o.reward_for_agent(self.dp.agent_to_act(s)), w);
            prev = n;
        }
        // TODO: see a way to update root node info, as it is not propogated. Also, do we need that?
    }

    pub(crate) fn once(&self, n: NS::NodeRef, s: &mut D::State)
    where
        P: MoveProcessor<D, I>,
    {
        let mut stack = vec![];
        let node = self.select(n, s, &mut stack);
        let (outcome_opt, generated_moves, terminal) =
            self.preprocessor.generate_moves(&self.dp, s);
        if terminal {
            node.mark_solved();
        } else {
            self.store.create_outgoing_edges(&node, generated_moves);
        }
        let outcome = outcome_opt
            .or(Some(self.simulator.sample_outcome(&self.dp, s)))
            .unwrap();
        self.backtrack(node, s, &mut stack, outcome, 1);
    }

    // Ensure that there is a certian number of visitations on the node n before attempting to
    // run a block, as otherwise, it could cause many collisions, which are not handled as of now.
    fn one_block(&self, n: NS::NodeRef, s: &mut D::State, block: usize)
    where
        P: BlockMoveProcessor<D, I>,
        D::State: Clone,
    {
        let mut states = Vec::with_capacity(block);
        let mut nodes_and_stacks = Vec::with_capacity(block);
        for _ in 0..block {
            let mut local_s = s.clone();
            let mut local_stack = vec![];
            let node = self.select(n.clone(), &mut local_s, &mut local_stack);
            // TODO: short/cache on states which do not need nnet expansion
            states.push(local_s);
            nodes_and_stacks.push((node, local_stack));
        }
        let expansion_results = self.preprocessor.generate_moves(&self.dp, &mut states);
        for (((outcome_opt, generated_edges, terminate), (node, mut stack)), mut state) in
            expansion_results
                .into_iter()
                .zip(nodes_and_stacks.into_iter())
                .zip(states.into_iter())
        {
            if terminate {
                node.mark_solved();
            } else {
                self.store.create_outgoing_edges(&node, generated_edges);
            }
            let outcome = outcome_opt
                .or(Some(self.simulator.sample_outcome(&self.dp, &mut state)))
                .unwrap();
            self.backtrack(node, &mut state, &mut stack, outcome, 1);
        }
    }

    pub(crate) fn ensure_valid_starting_node(&self, n: NS::NodeRef, s: &mut D::State)
    where
        P: MoveProcessor<D, I>,
    {
        if !n.is_finished_expanding() && !n.is_solved() {
            // This is to ensure that the invariant is maintained at all nodes (sum of selection
            // count of children + 1 = selection count of node (because of expansion))
            n.increment_selection_count();
            let (outcome_opt, generated_moves, terminal) =
                self.preprocessor.generate_moves(&self.dp, s);
            if terminal {
                n.mark_solved();
            } else {
                self.store.create_outgoing_edges(&n, generated_moves);
            }
            let outcome = outcome_opt
                .or(Some(self.simulator.sample_outcome(&self.dp, s)))
                .unwrap();
            self.backtrack(n, s, &mut vec![], outcome, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::graph_dp::tests::{problem1, problem2, DSim};
    use crate::lib::decision_process::DefaultSimulator;
    use crate::lib::mcts::node_store::{ActionWithStaticPolicy, OnlyAction};
    use crate::lib::mcts::safe_tree::tests::print_tree;
    use crate::lib::mcts::safe_tree::ThreadSafeNodeStore;
    use crate::lib::mcts::tree_policy::{
        PuctTreePolicy, PuctWithDiricheletTreePolicy, RandomTreePolicy, UctTreePolicy,
    };
    use crate::lib::{NoFilteringAndUniformPolicyForPuct, NoProcessing};

    #[test]
    fn test_basic() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            RandomTreePolicy,
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        let mut stack = vec![];
        for i in 0..10 {
            assert_eq!(node.total_selection_count(), i + 1);
            let k4 = s.select(node.clone(), &mut state, &mut stack);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_basic_2() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            RandomTreePolicy,
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..10 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_block() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            RandomTreePolicy,
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        s.one_block(node.clone(), &mut state, 4);
        print_tree(s.store(), &node);
    }

    #[test]
    fn test_uct_2() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            UctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..100 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_uct_block() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            UctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for _ in 0..5 {
            s.one_block(node.clone(), &mut state, 50);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_2() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            PuctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..100 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_block() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            PuctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for _ in 0..5 {
            s.one_block(node.clone(), &mut state, 50);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_d_2() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            PuctWithDiricheletTreePolicy::new(2.4, 1.8, 0.25),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..100 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_d_block() {
        let s = Search::new(
            problem1(),
            DefaultSimulator,
            PuctWithDiricheletTreePolicy::new(2.4, 1.8, 0.25),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for _ in 0..5 {
            s.one_block(node.clone(), &mut state, 50);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_2_multi_agent() {
        let s = Search::new(
            problem2(),
            DSim,
            PuctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..250 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
            print_tree(s.store(), &node);
        }
    }

    #[test]
    fn test_puct_2_multi_agent_block() {
        let s = Search::new(
            problem2(),
            DSim,
            PuctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for _ in 0..5 {
            s.one_block(node.clone(), &mut state, 50);
            print_tree(s.store(), &node);
        }
    }
}

use crate::lib::decision_process::DecisionProcess;
use petgraph::prelude::*;
use std::collections::BTreeMap;

type Action = u32;
type Outcome = Vec<f32>;
type Agent = u32;

pub(crate) struct GraphDP {
    start_state: NodeIndex,
    // for each state, we need to store the agent to move
    graph: Graph<Agent, Action>,
    terminal_states: BTreeMap<NodeIndex, Outcome>,
}

impl DecisionProcess for GraphDP {
    type Agent = Agent;
    type Action = Action;
    // We just store the state to return to as the undo_action as our state's are copy
    type UndoAction = NodeIndex;
    type State = NodeIndex;
    type Outcome = Outcome;

    fn start_state(&self) -> Self::State {
        self.start_state
    }

    type Actions = <Vec<Action> as IntoIterator>::IntoIter;

    fn legal_actions(&self, s: &Self::State) -> Self::Actions {
        self.graph
            .edges(*s)
            .map(|w| *w.weight())
            .collect::<Vec<Action>>()
            .into_iter()
    }

    fn agent_to_act(&self, s: &Self::State) -> Self::Agent {
        *self.graph.node_weight(*s).unwrap()
    }

    fn transition(&self, s: &mut Self::State, a: &Self::Action) -> Self::UndoAction {
        let old_state = *s;
        for edge in self.graph.edges(*s) {
            if *a == *edge.weight() {
                *s = edge.target();
                break;
            }
        }
        old_state
    }

    fn undo_transition(&self, s: &mut Self::State, u: Self::UndoAction) {
        *s = u;
    }

    fn is_finished(&self, s: &Self::State) -> Option<Self::Outcome> {
        self.terminal_states.get(s).map(|x| x.clone())
    }
}

impl crate::lib::decision_process::Outcome<Agent> for Vec<f32> {
    type RewardType = f32;

    fn reward_for_agent(&self, a: u32) -> Self::RewardType {
        match a {
            0 => self[0],
            1 => {
                if self.len() == 1 {
                    -self[0]
                } else {
                    self[1]
                }
            }
            _ => panic!("vdcdsvc"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::lib::decision_process::Simulator;

    /// n0
    /// +--> 1 ->> n1 (1)
    /// +--> 2 ->> n2 (2)
    /// +--> 3 ->> n3 (3)
    pub(crate) fn problem1() -> GraphDP {
        let mut result = GraphDP {
            start_state: Default::default(),
            graph: Graph::new(),
            terminal_states: Default::default(),
        };
        let mut g = &mut result.graph;
        // all nodes have the same agent
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        let n2 = g.add_node(0);
        let n3 = g.add_node(0);

        g.add_edge(n0, n1, 1);
        g.add_edge(n0, n2, 2);
        g.add_edge(n0, n3, 3);
        result.terminal_states.insert(n1, vec![1.0]);
        result.terminal_states.insert(n2, vec![2.0]);
        result.terminal_states.insert(n3, vec![3.0]);
        result
    }

    pub(crate) struct DSim;
    impl Simulator<GraphDP> for DSim {
        fn sample_outcome(
            &self,
            d: &GraphDP,
            state: &mut <GraphDP as DecisionProcess>::State,
        ) -> <GraphDP as DecisionProcess>::Outcome {
            vec![0.0, 0.0]
        }
    }

    pub(crate) fn problem2() -> GraphDP {
        let mut result = GraphDP {
            start_state: Default::default(),
            graph: Graph::new(),
            terminal_states: Default::default(),
        };
        let mut g = &mut result.graph;
        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(1);
        let n3 = g.add_node(0);
        let n4 = g.add_node(0);

        g.add_edge(n0, n1, 0);
        g.add_edge(n0, n2, 1);
        g.add_edge(n1, n3, 2);
        g.add_edge(n1, n4, 3);
        result.terminal_states.insert(n2, vec![1.0, -1.0]);
        result.terminal_states.insert(n3, vec![0.0, 0.0]);
        result.terminal_states.insert(n4, vec![-1.0, 1.0]);
        result
    }
}

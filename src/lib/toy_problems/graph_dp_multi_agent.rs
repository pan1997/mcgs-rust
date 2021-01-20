use crate::lib::decision_process::DecisionProcess;
use petgraph::prelude::*;
use std::collections::BTreeMap;

type State = NodeIndex;
type Action = u32;
type Outcome = Vec<u32>;
type Agent = u32;

struct GraphDP {
    start_state: State,
    // for each state, we need to store the agent to move
    graph: Graph<Agent, Action>,
    terminal_states: BTreeMap<State, Outcome>,
}

impl DecisionProcess for GraphDP {
    type Agent = Agent;
    type Action = Action;
    // We just store the state to return to as the undo_action as our state's are copy
    type UndoAction = State;
    type State = State;
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

impl crate::lib::decision_process::Outcome<Agent> for Vec<u32> {
    type RewardType = u32;

    fn reward_for_agent(&self, a: u32) -> Self::RewardType {
        *self.get(a as usize).unwrap()
    }
}

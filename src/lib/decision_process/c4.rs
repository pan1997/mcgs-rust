use crate::lib::decision_process::{DecisionProcess, Outcome};
use std::fmt::{Display, Formatter};
use std::num::Wrapping;

type Player = i8;

const B: Player = i8::MIN;
const W: Player = i8::MAX;

pub struct Board {
    white_pieces: Vec<bool>,
    black_pieces: Vec<bool>,
    height: Vec<u8>,
    player_to_move: Player,

    last_move: Move,
    total_piece_count: usize,
}

// The enclosed value is the square column
#[derive(Clone, Copy)]
pub struct Move(pub u8);

pub(crate) struct C4 {
    width: usize,
    height: usize,
    size: usize, // = width * height
}
// 0 is draw, otherwise it stores the winner
//pub struct C4Outcome(Player);

impl DecisionProcess for C4 {
    type Agent = Player;
    type Action = Move;
    type UndoAction = Move;
    type State = Board;

    // winning percentage for white, normalised between [-1, 1]
    type Outcome = f32;

    fn start_state(&self) -> Self::State {
        Board {
            white_pieces: vec![false; self.size],
            black_pieces: vec![false; self.size],
            height: vec![0; self.width],
            player_to_move: W,
            last_move: Move(u8::MAX), // invalid move
            total_piece_count: 0,
        }
    }

    type Actions = <Vec<Move> as IntoIterator>::IntoIter;

    fn legal_actions(&self, s: &Self::State) -> Self::Actions {
        let buf: Vec<Move> = (0..self.width)
            .filter(|c| (s.height[*c] as usize) < self.height)
            .map(|c| Move(c as u8))
            .collect();
        buf.into_iter()
    }

    fn agent_to_act(&self, s: &Self::State) -> Self::Agent {
        s.player_to_move
    }

    fn transition(&self, s: &mut Self::State, a: &Self::Action) -> Self::UndoAction {
        let column = a.0 as usize;
        debug_assert!(column < s.height.len());
        let b = match s.player_to_move {
            W => &mut s.white_pieces,
            _ => &mut s.black_pieces,
        };
        let height = &mut s.height[column as usize];
        b[self.translate(*height as usize, column)] = true;
        *height += 1;
        let undo = s.last_move;
        s.last_move = *a;
        s.player_to_move = opponent(s.player_to_move);
        s.total_piece_count += 1;
        undo
    }

    fn undo_transition(&self, s: &mut Self::State, u: Self::UndoAction) {
        s.total_piece_count -= 1;
        s.player_to_move = opponent(s.player_to_move);
        let undo_move = s.last_move;
        s.last_move = u;
        let column = undo_move.0 as usize;
        debug_assert!(column < s.height.len());
        let height = &mut s.height[column as usize];
        *height -= 1;
        let b = match s.player_to_move {
            W => &mut s.white_pieces,
            _ => &mut s.black_pieces,
        };
        b[self.translate(*height as usize, column)] = false;
    }

    fn is_finished(&self, s: &Self::State) -> Option<Self::Outcome> {
        if s.last_move.0 as usize > self.width {
            None
        } else {
            let player_who_just_moved = opponent(s.player_to_move);
            let board = match player_who_just_moved {
                W => &s.white_pieces,
                _ => &s.black_pieces,
            };
            let column = s.last_move.0 as usize;
            let row = (s.height[column] - 1) as usize;
            let ll = self.count_chain_length(board, row, column, 0, usize::MAX);
            let rr = self.count_chain_length(board, row, column, 0, 1);
            let dd = self.count_chain_length(board, row, column, usize::MAX, 0);
            let lu = self.count_chain_length(board, row, column, 1, usize::MAX);
            let ld = self.count_chain_length(board, row, column, usize::MAX, usize::MAX);
            let ru = self.count_chain_length(board, row, column, 1, 1);
            let rd = self.count_chain_length(board, row, column, usize::MAX, 1);
            /*println!(
                "ll: {}, rr: {}, dd: {}, lu: {}, ld: {}, ru: {}, rd: {}",
                ll, rr, dd, lu, ld, ru, rd
            );*/
            if dd >= 3 || ll + rr >= 3 || lu + rd >= 3 || ld + ru >= 3 {
                Some(match player_who_just_moved {
                    B => -1.0,
                    _ => 1.0,
                })
            } else if s.total_piece_count == self.size {
                Some(0.0)
            } else {
                None
            }
        }
    }
}

impl C4 {
    fn translate(&self, r: usize, c: usize) -> usize {
        if c < self.width && r < self.size {
            self.width * r + c
        } else {
            usize::MAX
        }
    }

    fn count_chain_length(&self, v: &Vec<bool>, r: usize, c: usize, dx: usize, dy: usize) -> usize {
        for i in 1..4 {
            let index = self.translate(
                (Wrapping(r) + Wrapping(dx) * Wrapping(i)).0,
                (Wrapping(c) + Wrapping(dy) * Wrapping(i)).0,
            );
            if index >= self.size || !v[index] {
                return i - 1;
            }
        }
        3
    }

    pub(crate) fn new(width: usize, height: usize) -> C4 {
        C4 {
            width,
            height,
            size: width * height,
        }
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let w = self.height.len();
        let h = self.white_pieces.len() / w;
        for row in 0..h {
            for col in 0..w {
                let index = w * (h - row - 1) + col;
                if self.white_pieces[index] {
                    write!(f, "W|")?;
                } else if self.black_pieces[index] {
                    write!(f, "B|")?;
                } else {
                    write!(f, " |")?;
                }
            }
            writeln!(f)?;
        }
        for col in 0..w {
            write!(f, "{}|", col % 10)?;
        }
        writeln!(f)
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn opponent(p: Player) -> Player {
    !p
}

impl Outcome<Player> for f32 {
    type RewardType = f32;

    fn reward_for_agent(&self, a: Player) -> Self::RewardType {
        match a {
            B => -*self,
            _ => *self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::{DefaultSimulator, RandomSimulator};
    use crate::lib::mcts::node_store::{ActionWithStaticPolicy, Node, NodeStore, OnlyAction};
    use crate::lib::mcts::safe_tree::get_total_simulation_counts;
    use crate::lib::mcts::safe_tree::tests::node_distribution;
    use crate::lib::mcts::safe_tree::tests::print_tree;
    use crate::lib::mcts::safe_tree::ThreadSafeNodeStore;
    use crate::lib::mcts::tree_policy::{
        PuctTreePolicy, PuctWithDiricheletTreePolicy, UctTreePolicy,
    };
    use crate::lib::mcts::Search;
    use crate::lib::{NoFilteringAndUniformPolicyForPuct, NoProcessing};

    #[test]
    fn basic() {
        assert_eq!(opponent(B), W);

        assert_eq!(0.0.reward_for_agent(W), 0.0);
        assert_eq!(0.0.reward_for_agent(B), 0.0);
        assert_eq!(1.0.reward_for_agent(W), 1.0);
        assert_eq!(1.0.reward_for_agent(B), -1.0);
        assert_eq!(-1.0.reward_for_agent(W), -1.0);
        assert_eq!(-1.0.reward_for_agent(B), 1.0);
    }

    #[test]
    fn test_moves() {
        let c4 = C4::new(7, 6);
        let b = &mut c4.start_state();
        println!("{}", b);
        //let moves = vec![4, 4, 3, 3, 5, 5, 6];
        let moves = vec![4, 3, 3, 2, 2, 1, 2, 1, 1];
        for col in moves {
            assert!(c4.is_finished(b).is_none());
            c4.transition(b, &Move(col));
            println!("{}", b);
        }
        //assert_eq!(c4.is_finished(b).unwrap().0, W);
    }

    #[test]
    fn mcts_test() {
        let s = Search::new(
            C4::new(7, 6),
            RandomSimulator,
            UctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<OnlyAction<_>>::new(),
            NoProcessing,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..20000 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
        }
        node_distribution(s.store(), &node);
        assert_eq!(
            get_total_simulation_counts(s.store(), node.clone()) + 1,
            node.total_selection_count()
        );
        s.print_pv(node, None, None, 20);
    }

    #[test]
    fn mcts_test_puct() {
        let s = Search::new(
            C4::new(7, 6),
            RandomSimulator,
            PuctTreePolicy::new(2.4),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..20000 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
        }
        node_distribution(s.store(), &node);
    }

    #[test]
    fn mcts_test_puct_w() {
        let s = Search::new(
            C4::new(7, 6),
            RandomSimulator,
            PuctWithDiricheletTreePolicy::new(1.2, 1.8, 0.25),
            ThreadSafeNodeStore::<ActionWithStaticPolicy<_>>::new(),
            NoFilteringAndUniformPolicyForPuct,
        );

        let node = s.store().new_node();
        let mut state = s.dp().start_state();
        print_tree(s.store(), &node);
        s.ensure_valid_starting_node(node.clone(), &mut state);
        print_tree(s.store(), &node);
        for i in 0..20000 {
            assert_eq!(node.total_selection_count(), i + 1);
            s.once(node.clone(), &mut state);
        }
        node_distribution(s.store(), &node);
    }
}

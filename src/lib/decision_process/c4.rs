use crate::lib::decision_process::{ComparableOutcome, DecisionProcess, Outcome, WinnableOutcome};
use crate::lib::mcgs::graph::Hsh;
use rand::{thread_rng, Rng};
use std::fmt::{Display, Formatter};
use std::num::Wrapping;
use tch::Tensor;

type Player = i8;

const B: Player = i8::MIN;
const W: Player = i8::MAX;

#[derive(Clone)]
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

pub struct C4 {
    width: usize,
    height: usize,
    size: usize, // = width * height
}

impl DecisionProcess for C4 {
    type Agent = Player;
    type Action = Move;
    type UndoAction = Move;
    type State = Board;

    // winning percentage for white, normalised between [-1, 1]
    type Outcome = f64;

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
    pub fn width(&self) -> usize {
        self.width
    }

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

    pub fn new(width: usize, height: usize) -> C4 {
        C4 {
            width,
            height,
            size: width * height,
        }
    }

    pub(crate) fn generate_tensor(&self, state: &Board) -> Tensor {
        let dim = [self.height as i64, self.width as i64];
        let white = Tensor::of_slice(&state.white_pieces).reshape(&dim);
        let black = Tensor::of_slice(&state.black_pieces).reshape(&dim);
        let empty = white.logical_or(&black).logical_not();
        match state.player_to_move {
            B => Tensor::stack(&[black, white, empty], 0).unsqueeze(0),
            _ => Tensor::stack(&[white, black, empty], 0).unsqueeze(0),
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

impl Outcome<Player> for f64 {
    type RewardType = f32;

    fn reward_for_agent(&self, a: Player) -> Self::RewardType {
        match a {
            B => -*self as f32,
            _ => *self as f32,
        }
    }
}

impl WinnableOutcome<Player> for f32 {
    fn is_winning_for(&self, a: Player) -> bool {
        match a {
            B => *self < -0.95,
            _ => *self > 0.95,
        }
    }

    fn is_losing_for(&self, a: i8) -> bool {
        self.is_winning_for(opponent(a))
    }
}

impl WinnableOutcome<Player> for f64 {
    fn is_winning_for(&self, a: Player) -> bool {
        match a {
            B => *self < -0.95,
            _ => *self > 0.95,
        }
    }

    fn is_losing_for(&self, a: i8) -> bool {
        self.is_winning_for(opponent(a))
    }
}

impl ComparableOutcome<Player> for f32 {
    fn is_better_than(&self, other: &Self, a: i8) -> bool {
        self.reward_for_agent(a) > other.reward_for_agent(a)
    }
}

impl ComparableOutcome<Player> for f64 {
    fn is_better_than(&self, other: &Self, a: i8) -> bool {
        self.reward_for_agent(a) > other.reward_for_agent(a)
    }
}

pub struct ZobHash {
    keys_w: Vec<u64>,
    keys_b: Vec<u64>,
}

impl ZobHash {
    pub fn new(l: usize) -> Self {
        ZobHash {
            keys_w: (0..l).map(|_| thread_rng().gen()).collect(),
            keys_b: (0..l).map(|_| thread_rng().gen()).collect(),
        }
    }
}

impl Hsh<Board> for ZobHash {
    type K = u64;

    fn key(&self, s: &Board) -> Self::K {
        let mut ans = 0;
        for index in 0..self.keys_b.len() {
            if s.white_pieces[index] {
                ans ^= self.keys_w[index]
            } else if s.black_pieces[index] {
                ans ^= self.keys_b[index]
            }
        }
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::RandomSimulator;
    use crate::lib::{
        ActionWithStaticPolicy, NoFilteringAndUniformPolicyForPuct, NoProcessing, OnlyAction,
    };

    #[test]
    fn tensor_test() {
        let c4 = C4::new(7, 6);
        let b = &mut c4.start_state();
        println!("{}", b);
        let moves = vec![4, 3, 3, 2, 2, 1, 2, 1, 1];
        for col in moves {
            assert!(c4.is_finished(b).is_none());
            c4.transition(b, &Move(col));
            println!("{}", b);
        }
        let u = c4.generate_tensor(b);
        u.print();
    }
}

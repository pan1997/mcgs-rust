use crate::lib::decision_process::DecisionProcess;
use std::fmt::{Display, Formatter};

type Player = i8;

const B: Player = i8::MIN;
const W: Player = i8::MAX;

#[derive(Clone)]
pub struct Board {
    white_rows: Vec<u32>,
    black_columns: Vec<u32>,
    player_to_move: Player,
}

impl Board {
    fn new(height: usize, width: usize) -> Self {
        Board {
            white_rows: vec![0; height],
            black_columns: vec![0; width],
            player_to_move: W,
        }
    }
}

pub struct Hex {
    width: u32,
    height: u32,
}

impl Hex {
    pub fn new(height: u32, width: u32) -> Self {
        Hex { width, height }
    }

    fn count(pieces: &Vec<u32>) -> usize {
        //println!("Counting");
        let mut reachable = vec![0; pieces.len()];
        reachable[0] = pieces[0];
        let mut cursor = 0;
        loop {
            let mut row = reachable[cursor];
            //println!("Cursor: {}, row: {:011b}", cursor, row);
            if row == 0 {
                return cursor
            }

            let mut row_prev = row;
            // expand row horizontally
            loop {
                let nr = ((row << 1) | (row >> 1) | row) & pieces[cursor];
                if nr == row {
                    break;
                }
                row = nr;
            }

            reachable[cursor] = row;
            let mut flag = true;
            if row != row_prev {
                //println!("Cursor: {}, row: {:011b}", cursor, row);
                while (cursor > 0) {
                    cursor -= 1;
                    let new = (reachable[cursor] | (row << 1) | row) & pieces[cursor];
                    if new != reachable[cursor] {
                        reachable[cursor] = new;
                        row = new;
                        flag = false;
                    } else {
                        cursor += 1;
                        break;
                    }
                }
            }

            if flag {
                if cursor + 1 == pieces.len() {
                    return cursor + 1
                }
                let next_row = (row | (row >> 1)) & pieces[cursor + 1];
                cursor += 1;
                reachable[cursor] = next_row
            }
        }
    }

    fn flip(pieces: &mut u32, index: u32) {
        *pieces ^= (1 << index);
    }

    fn get(pieces: u32, index: u32) -> u32 {
        (pieces >> index) & 1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Move(pub u16, pub u16);

impl DecisionProcess for Hex {
    type Agent = Player;
    type Action = Move;
    type UndoAction = Move;
    type State = Board;
    type Outcome = f64;

    fn start_state(&self) -> Self::State {
        Board::new(self.height as usize, self.width as usize)
    }

    type Actions = <Vec<Move> as IntoIterator>::IntoIter;

    fn legal_actions(&self, s: &Self::State) -> Self::Actions {
        let mut a = vec![];
        for row in 0..self.height {
            for col in 0..self.width {
                if Hex::get(s.white_rows[row as usize], col) == 0
                    && Hex::get(s.black_columns[col as usize], row) == 0
                {
                    a.push(Move(row as u16, col as u16))
                }
            }
        }
        if a.len() == 0 {
            println!("{}", s);
            println!("{:?} {:?}", s.white_rows, s.black_columns);
            panic!("cannot return an empty action list");
        }
        a.into_iter()
    }

    fn agent_to_act(&self, s: &Self::State) -> Self::Agent {
        s.player_to_move
    }

    fn transition(&self, s: &mut Self::State, a: &Self::Action) -> Self::UndoAction {
        match s.player_to_move {
            B => Hex::flip(&mut s.black_columns[a.1 as usize], a.0 as u32),
            W => Hex::flip(&mut s.white_rows[a.0 as usize], a.1 as u32),
            _ => panic!("illegal player"),
        }
        s.player_to_move = opponent(s.player_to_move);
        *a
    }

    fn undo_transition(&self, s: &mut Self::State, u: Self::UndoAction) {
        s.player_to_move = opponent(s.player_to_move);
        match s.player_to_move {
            B => Hex::flip(&mut s.black_columns[u.1 as usize], u.0 as u32),
            W => Hex::flip(&mut s.white_rows[u.0 as usize], u.1 as u32),
            _ => panic!("illegal player"),
        }
    }

    fn is_finished(&self, s: &Self::State) -> Option<Self::Outcome> {
        // These need to be opposite, as the player who just moved is different from the current
        // player
        match s.player_to_move {
            B => if Hex::count(&s.white_rows) == self.height as usize {
                Some(1.0)
            } else {
                None
            }
            W => if Hex::count(&s.black_columns) == self.width as usize {
                Some(-1.0)
            } else {
                None
            }
            _ => panic!("illegal player"),
        }
    }
}

fn opponent(p: Player) -> Player {
    !p
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let row_count = self.white_rows.len();
        let col_count = self.black_columns.len();
        for row in 0..row_count {
            for _ in 0..row {
                write!(f, " ")?;
            }
            write!(f, "|")?;
            for col in 0..col_count {
                if Hex::get(self.white_rows[row], col as u32) == 1 {
                    write!(f, "X|")?;
                } else if Hex::get(self.black_columns[col], row as u32) == 1 {
                    write!(f, "O|")?;
                } else {
                    write!(f, " |")?;
                }
            }
            writeln!(f)?;
        }
        if self.player_to_move == B {
            writeln!(f, "B to move")
        } else {
            writeln!(f, "W to move")
        }
    }
}
impl Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lib::decision_process::DecisionProcess;

    #[test]
    fn t1() {
        let game = Hex::new(11, 11);
        let b = &mut game.start_state();
        println!("{}", b);
        assert!(game.is_finished(b).is_none());
        let actions: Vec<Move> = game.legal_actions(b).collect();
        let action = actions[4];
        let u = game.transition(b, &action);
        println!("{}", b);
        game.undo_transition(b, u);
        println!("{}", b);
    }

    #[test]
    fn t2() {
        let h1 = vec![1901, 51, 47, 450, 421, 347, 293, 1067, 309, 1192, 2047];
        let h2= vec![520, 849, 682, 346, 733, 40, 982, 487, 646, 1022, 382];
        let game = Hex::new(11, 11);
        let mut b = Board {
            white_rows: h1,
            black_columns: h2,
            player_to_move: B
        };
        println!("{}", b);
        for x in b.white_rows.iter() {
            println!("{:011b}", x);
        }
        println!("{}", Hex::count(&b.white_rows));
        let actions: Vec<Move> = game.legal_actions(&b).collect();
        println!("{:?}", actions);
    }

}
//! An implementation of Negamax.
//!
//! With only the basic alpha-pruning implemented. This picks randomly among
//! the "best" moves, so that it's non-deterministic.

use std::cmp::max;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::time_it_macro;

use minimax::interface::*;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;

#[derive(Debug, Clone)]
pub(crate) struct MovePool<M> {
    pool: Vec<Vec<M>>,
}

pub struct Negamax<E: Evaluator> {
    max_depth: usize,
    move_pool: MovePool<<E::G as Game>::M>,
    rng: ThreadRng,
    prev_value: Evaluation,
    eval: E,
}

impl<M> Default for MovePool<M> {
    fn default() -> Self {
        Self { pool: Vec::new() }
    }
}

impl<M> MovePool<M> {
    pub(crate) fn alloc(&mut self) -> Vec<M> {
        self.pool.pop().unwrap_or_else(Vec::new)
    }

    pub(crate) fn free(&mut self, mut vec: Vec<M>) {
        vec.clear();
        self.pool.push(vec);
    }
}

impl<E: Evaluator> Negamax<E> {
    pub fn new(eval: E, depth: usize) -> Negamax<E> {
        Negamax {
            max_depth: depth,
            move_pool: MovePool::<_>::default(),
            rng: rand::thread_rng(),
            prev_value: 0,
            eval,
        }
    }

    fn negamax(
        &mut self, s: &mut <E::G as Game>::S, depth: usize, mut alpha: Evaluation, beta: Evaluation,
    ) -> Evaluation {
        // static COUNTER: AtomicUsize = AtomicUsize::new(0);
        if let Some(winner) = E::G::get_winner(s) {
            return winner.evaluate();
        }
        if depth == 0 {
            return self.eval.evaluate(s);
        }
        let mut moves = self.move_pool.alloc();
        E::G::generate_moves(s, &mut moves);
        let mut best = WORST_EVAL;
        // println!("Moves size: {}", moves.len());
        time_it_macro!("iteration", {
            for m in moves.iter() {
                m.apply(s);
                let value = -self.negamax(s, depth - 1, -beta, -alpha);
                m.undo(s);
                best = max(best, value);
                alpha = max(alpha, value);
                if alpha >= beta {
                    break;
                }
            }
        });
        self.move_pool.free(moves);
        // COUNTER.fetch_add(1, Ordering::Relaxed);
        // println!("Counter: {:?}", COUNTER);
        best
    }
}

impl<E: Evaluator> Strategy<E::G> for Negamax<E>
    where
        <E::G as Game>::S: Clone,
        <E::G as Game>::M: Clone,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        let mut best = WORST_EVAL;
        let mut moves = self.move_pool.alloc();
        E::G::generate_moves(s, &mut moves);
        // Randomly permute order that we look at the moves.
        // We'll pick the first best score from this list.
        moves[..].shuffle(&mut self.rng);

        let mut best_move = moves.first().cloned()?;
        let mut s_clone = s.clone();
        for m in &moves {
            // determine value for this move
            m.apply(&mut s_clone);
            let value =
                time_it_macro!("rec", { -self.negamax(&mut s_clone, self.max_depth, WORST_EVAL, -best) });
            m.undo(&mut s_clone);
            // Strictly better than any move found so far.
            if value > best {
                best = value;
                best_move = m.clone();
            }
        }
        self.move_pool.free(moves);
        self.prev_value = best;
        Some(best_move.clone())
    }
}

//! Shared game setup used by training, benchmarking, and tests.

use std::cell::RefCell;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::{SmallRng, StdRng};


use duke_rust::game::ai::player::{AiMove, EvaluatingPlayer};
use duke_rust::game::ai::player::ArtificialPlayer;
use duke_rust::game::ai::stupid_sync_ai::StupidSyncAi;
use duke_rust::game::bag::TileBag;
use duke_rust::game::board::{DukeOffset, PossibleMove};
use duke_rust::game::board_setup::{DukeInitialLocation, FootmenSetup};
use duke_rust::game::state::{GameMove, GameResult, GameState};
use duke_rust::game::tile::{Owner, TileType};

use duke_rust::game::ai::heuristics::Heuristic;

use crate::encoding::{active_board_features, bag_features};
use crate::generic_mlp::{GenericMlp, L1Accumulator};
use crate::learned_heuristic::{extract_combined_features, NUM_COMBINED_FEATURES};
use crate::nnue::NnueEvaluator;

thread_local! {
    /// Reused candidate list for greedy paths (single-thread sequential use per call).
    static GREEDY_CANDIDATES: RefCell<Vec<PossibleMove>> = RefCell::new(Vec::new());
    /// One move list per search ply so nested `negamax_ab` does not clobber parent buffers.
    static NEGAMAX_MOVE_POOL: RefCell<[Vec<PossibleMove>; 64]> =
        RefCell::new(std::array::from_fn(|_| Vec::new()));
}

/// Piece moves without guard, then up to four guard-checked placements (matches `greedy_move` semantics).
fn collect_greedy_candidate_moves(gs: &mut GameState, owner: Owner, out: &mut Vec<PossibleMove>) {
    out.clear();
    gs.all_valid_tile_moves_ignoring_guard_into(owner, out);
    if gs.bag_for_current_player().non_empty() {
        for &offset in &[DukeOffset::Top, DukeOffset::Bottom, DukeOffset::Left, DukeOffset::Right] {
            if gs.is_valid_placement(offset) {
                out.push(PossibleMove::PlaceNewTile(offset, owner));
            }
        }
    }
}

/// Safety limit: if a game exceeds this many turns, force a draw.
/// In practice the built-in idle-move draw rule should trigger well before this.
pub const MAX_TURNS: u32 = 500;

/// Terminal game scores for minimax / labeling.
/// Using ±30 keeps terminal values in the same ballpark as heuristic evaluations
/// (which typically range from roughly −20 to +20), so the search doesn't
/// over-weight shallow forced wins relative to positional advantages.
pub const TERMINAL_WIN_SCORE: f64 = 30.0;
pub const TERMINAL_LOSS_SCORE: f64 = -30.0;

/// Convert a game result to a training target from the perspective of `current_player`.
///
/// Returns `Some(1.0)` for a win, `Some(-1.0)` for a loss, `Some(0.0)` for a tie,
/// and `None` for an ongoing game (caller decides whether to skip or use 0.0).
pub fn game_result_target(result: GameResult, current_player: Owner) -> Option<f64> {
    match result {
        GameResult::Won(w) if w == current_player => Some(1.0),
        GameResult::Won(_) => Some(-1.0),
        GameResult::Tie => Some(0.0),
        GameResult::Ongoing => None,
    }
}

/// Common trait for anything that can evaluate a game state.
/// Higher values = better for the current player.
/// The scale is arbitrary -- only relative ordering matters for move selection.
pub trait GameEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32;

    /// If this evaluator wraps a `GenericMlp` with sparse NNUE-style inputs,
    /// return a reference to the network and whether combined features are
    /// included (true for 1147-input, false for 1106-input models).
    ///
    /// Used by `greedy_move` to enable the incremental L1 accumulator path.
    /// Default implementation returns `None` (no accumulator support).
    fn as_generic_mlp(&self) -> Option<(&GenericMlp, bool)> { None }

    /// Whether this evaluator supports incremental heuristic evaluation.
    fn is_static_heuristic(&self) -> bool { false }
}

impl GameEvaluator for NnueEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.evaluate_state(gs)
    }
}

/// Wrapper that adapts an `EvaluatingPlayer` (heuristic) to the `GameEvaluator` trait.
pub struct HeuristicEvaluator<'a> {
    inner: &'a dyn EvaluatingPlayer,
}

impl<'a> HeuristicEvaluator<'a> {
    pub fn new(inner: &'a dyn EvaluatingPlayer) -> Self {
        Self { inner }
    }
}

impl GameEvaluator for HeuristicEvaluator<'_> {
    fn evaluate(&self, gs: &GameState) -> f32 {
        self.inner.cheap_evaluate(gs) as f32
    }
}

/// Static heuristic evaluator using the four known heuristic enum values.
/// Fully `Send + Sync` -- no trait objects, works with rayon.
pub struct StaticHeuristicEvaluator {
    heuristics: Vec<duke_rust::game::ai::heuristics::Heuristics>,
}

impl StaticHeuristicEvaluator {
    pub fn new() -> Self {
        use duke_rust::game::ai::heuristics::Heuristics;
        Self {
            heuristics: vec![
                Heuristics::DukeMovementOptions,
                Heuristics::TotalTilesOnBoard,
                Heuristics::TotalMovementOptions,
                Heuristics::DiscardedUnits,
            ],
        }
    }
}

impl GameEvaluator for StaticHeuristicEvaluator {
    fn evaluate(&self, gs: &GameState) -> f32 {
        let owner = gs.current_player_turn();
        let (own_total, own_duke, own_tiles, opp_total, opp_duke, opp_tiles) =
            gs.heuristic_counts_both_players(owner);
        let duke_diff = own_duke as f64 - opp_duke as f64;
        let tiles_diff = 10.0 * (own_tiles as f64 - opp_tiles as f64);
        let moves_diff = own_total as f64 - opp_total as f64;
        let discard_diff = -15.0 * (gs.discard_bag_for(owner).len() as f64
            - gs.discard_bag_for(owner.next_player()).len() as f64);
        (duke_diff + tiles_diff + moves_diff + discard_diff) as f32
    }

    fn is_static_heuristic(&self) -> bool { true }
}

/// Create the standard tile bag (all tiles except Assassin).
pub fn create_bag() -> TileBag {
    TileBag::new(vec![
        TileType::Footman,
        TileType::Bowman,
        TileType::Knight,
        TileType::Pikeman,
        TileType::Pikeman,
        TileType::Champion,
        TileType::Priest,
        TileType::Wizard,
        TileType::Dragoon,
        // Assassin excluded: JumpSlide not yet fully implemented in board logic
        TileType::General,
        TileType::Marshall,
        TileType::Longbowman,
    ])
}

/// Create a standard initial game state.
pub fn create_initial_state(bag: &TileBag) -> GameState {
    GameState::new(
        bag,
        (DukeInitialLocation::Left, FootmenSetup::Left),
        (DukeInitialLocation::Right, FootmenSetup::Right),
    )
}

/// Play a game using random moves, collecting states at each turn.
pub fn play_random_game(gs: &GameState, rng: &mut StdRng) -> (Vec<GameState>, GameResult) {
    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if states.len() as u32 >= MAX_TURNS {
                    eprintln!("WARNING: play_random_game exceeded {} turns, forcing draw", MAX_TURNS);
                    states.push(game.clone());
                    return (states, GameResult::Tie);
                }
                states.push(game.clone());
                ai.play_next_move(rng, &mut game);
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

/// Play a game using self-play with epsilon-greedy exploration.
///
/// Accepts any `GameEvaluator` (NNUE, heuristic, etc.) for move selection.
/// Both players use the same evaluator.
pub fn play_selfplay_game<E: GameEvaluator + ?Sized>(
    gs: &GameState,
    evaluator: &E,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    play_two_player_game(gs, evaluator, evaluator, rng, epsilon)
}

/// Play a game with two different evaluators (top_eval vs bottom_eval).
///
/// `top_eval` selects moves when it is TopPlayer's turn, `bottom_eval` when BottomPlayer's.
/// Epsilon-greedy: with probability `epsilon`, a random move is played instead.
pub fn play_two_player_game<E1: GameEvaluator + ?Sized, E2: GameEvaluator + ?Sized>(
    gs: &GameState,
    top_eval: &E1,
    bottom_eval: &E2,
    rng: &mut StdRng,
    epsilon: f64,
) -> (Vec<GameState>, GameResult) {
    use duke_rust::game::tile::Owner;

    let ai = StupidSyncAi {};
    let mut game = gs.clone();
    let mut states = Vec::new();

    loop {
        match game.game_result() {
            GameResult::Ongoing => {
                if states.len() as u32 >= MAX_TURNS {
                    states.push(game.clone());
                    return (states, GameResult::Tie);
                }
                states.push(game.clone());

                if rng.gen::<f64>() < epsilon {
                    ai.play_next_move(rng, &mut game);
                } else {
                    let current = game.current_player_turn();
                    let mv = match current {
                        Owner::TopPlayer => greedy_move(&mut game, top_eval, rng),
                        Owner::BottomPlayer => greedy_move(&mut game, bottom_eval, rng),
                    };
                    mv.play(&mut game, rng);
                }
            }
            result => {
                states.push(game.clone());
                return (states, result);
            }
        }
    }
}

/// Negamax search with alpha-beta pruning and expectimax for tile draws.
///
/// Returns a score from the perspective of the current player (higher = better).
/// Terminal positions are scored as ±[`TERMINAL_WIN_SCORE`] or 0 (tie).
/// At depth 0 or when no moves are available, returns the static evaluation.
///
/// **Tile-draw handling (expectimax):** when the current player can draw a
/// tile from the bag, the drawn tile is random.  Instead of treating each
/// placement move deterministically, we compute the expected value of the
/// "draw" option by averaging over all distinct tile types in the bag
/// (weighted by their count).  For each tile type, we take the max over
/// valid placement locations, then weight-average across tile types.
/// The final value is the best of "best piece move" and "expected draw value".
pub fn negamax<E: GameEvaluator + ?Sized>(
    gs: &mut GameState, evaluator: &E, depth: u32, rng: &mut impl Rng,
) -> f64 {
    negamax_ab(gs, evaluator, depth, f64::NEG_INFINITY, f64::INFINITY, rng, 0)
}

/// Negamax with alpha-beta pruning.
///
/// `alpha` is the best score the current player can guarantee so far.
/// `beta` is the best score the opponent can guarantee.
/// When `alpha >= beta`, the remaining moves are pruned (beta cutoff).
fn negamax_ab<E: GameEvaluator + ?Sized>(
    gs: &mut GameState, evaluator: &E, depth: u32,
    mut alpha: f64, beta: f64,
    rng: &mut impl Rng,
    rec_depth: usize,
) -> f64 {
    const MAX_NEGAMAX_REC_DEPTH: usize = 64;
    debug_assert!(rec_depth < MAX_NEGAMAX_REC_DEPTH, "negamax_ab exceeded move pool depth");
    if rec_depth >= MAX_NEGAMAX_REC_DEPTH {
        return evaluator.evaluate(gs) as f64;
    }
    // Check tie first (O(1)) before doing any move generation.
    if gs.is_tie() {
        return 0.0;
    }

    if depth == 0 {
        // At the leaf we must still check for terminal positions (e.g. a duke
        // was captured on the parent's move) because the heuristic evaluator
        // panics on positions where duke_coordinates() returns None.
        // Use game_result() here -- it's only called at leaves, not interior
        // nodes, so the cost is acceptable.
        match gs.game_result() {
            GameResult::Won(winner) => {
                return if winner == gs.current_player_turn() {
                    TERMINAL_WIN_SCORE
                } else {
                    TERMINAL_LOSS_SCORE
                };
            }
            GameResult::Tie => return 0.0,
            GameResult::Ongoing => {}
        }
        return evaluator.evaluate(gs) as f64;
    }

    // Generate moves with guard checking (pooled vec per search ply).
    NEGAMAX_MOVE_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let moves = &mut pool[rec_depth];
        moves.clear();
        gs.all_valid_game_moves_for_current_player_into(moves);
        if !moves.is_empty() {
            let mut capture_end = 0usize;
            for i in 0..moves.len() {
                if matches!(&moves[i], PossibleMove::ApplyNonCommandTileAction { capturing: Some(_), .. }) {
                    moves.swap(i, capture_end);
                    capture_end += 1;
                }
            }
        }
    });

    if NEGAMAX_MOVE_POOL.with(|pool| pool.borrow()[rec_depth].is_empty()) {
        return TERMINAL_LOSS_SCORE;
    }

    let owner = gs.current_player_turn();

    let mut placement_offsets: [Option<DukeOffset>; 4] = [None; 4];
    let mut n_placements = 0usize;

    let mut best = f64::NEG_INFINITY;
    let base_rng = SmallRng::seed_from_u64(0);
    let moves_len = NEGAMAX_MOVE_POOL.with(|pool| pool.borrow()[rec_depth].len());
    for i in 0..moves_len {
        let pm = NEGAMAX_MOVE_POOL.with(|pool| pool.borrow()[rec_depth][i].clone());
        match &pm {
            PossibleMove::PlaceNewTile(offset, _) => {
                let already = placement_offsets[..n_placements].iter().any(|o| *o == Some(*offset));
                if !already {
                    placement_offsets[n_placements] = Some(*offset);
                    n_placements += 1;
                }
            }
            PossibleMove::ApplyNonCommandTileAction { src, dst, .. } => {
                let game_move = GameMove::ApplyNonCommandTileAction { src: *src, dst: *dst };
                let undo = pm.clone();
                gs.make_a_move(game_move, &mut base_rng.clone());
                let score = -negamax_ab(gs, evaluator, depth - 1, -beta, -alpha, rng, rec_depth + 1);
                gs.undo(undo);
                if score > best {
                    best = score;
                }
                if best > alpha {
                    alpha = best;
                }
                if alpha >= beta {
                    break;
                }
            }
        }
    }

    // Expectimax for the "draw from bag" option.
    // Note: alpha-beta pruning does NOT apply across expectimax branches
    // because the draw is stochastic (we must evaluate all tile types to
    // compute the expected value). However, we can still prune within each
    // tile-type's placement search using the current alpha/beta window.
    if n_placements > 0 && best < beta {
        let bag = gs.bag_for_current_player().remaining();
        let total_tiles = bag.len() as f64;
        debug_assert!(total_tiles > 0.0);

        // Count distinct tile types and their frequencies.
        let mut tile_counts = [0usize; 13];
        for &tile in bag {
            tile_counts[tile.index()] += 1;
        }

        let tile_types = [
            TileType::Duke, TileType::Footman, TileType::Pikeman, TileType::Knight,
            TileType::Champion, TileType::Dragoon, TileType::Wizard, TileType::General,
            TileType::Marshall, TileType::Assassin, TileType::Priest, TileType::Bowman,
            TileType::Longbowman,
        ];

        let mut draw_value = 0.0;
        for &tile_type in &tile_types {
            let count = tile_counts[tile_type.index()];
            if count == 0 { continue; }
            let prob = count as f64 / total_tiles;

            let mut best_for_tile = f64::NEG_INFINITY;
            for &offset_opt in &placement_offsets[..n_placements] {
                let offset = offset_opt.unwrap();
                // Pull specific tile and place in-place, then undo.
                gs.pull_specific_tile_from_bag(tile_type);
                let undo = PossibleMove::PlaceNewTile(offset, owner);
                gs.make_a_move(GameMove::PlaceNewTile(offset), &mut base_rng.clone());
                let score = -negamax_ab(gs, evaluator, depth - 1, -beta, -alpha, rng, rec_depth + 1);
                gs.undo(undo);
                if score > best_for_tile {
                    best_for_tile = score;
                }
            }

            draw_value += prob * best_for_tile;
        }

        if draw_value > best {
            best = draw_value;
        }
    }

    best
}

/// Arbitrary-depth minimax move selection using negamax.
///
/// Enumerates all legal moves, scores each via `negamax` at `depth - 1`,
/// and returns the move with the highest score.  Moves are shuffled before
/// evaluation so ties are broken randomly.
///
/// Works with any `GameEvaluator` implementation.  Terminal positions
/// (win/loss/tie) are handled explicitly so the search never misses a
/// forced win or avoids a forced loss.
pub fn greedy_move_deep<E: GameEvaluator + ?Sized>(
    gs: &mut GameState, evaluator: &E, depth: u32, rng: &mut impl Rng,
) -> AiMove {
    let (best_move, _) = greedy_move_deep_with_score(gs, evaluator, depth, rng);
    best_move
}

/// Same as greedy_move_deep but also returns the best score (from current player's perspective).
///
/// Uses make/undo in-place instead of cloning GameState per candidate move.
pub fn greedy_move_deep_with_score<E: GameEvaluator + ?Sized>(
    gs: &mut GameState, evaluator: &E, depth: u32, rng: &mut impl Rng,
) -> (AiMove, f64) {
    assert!(depth >= 1, "greedy_move_deep_with_score requires depth >= 1");
    GREEDY_CANDIDATES.with(|cell| {
        let mut moves = cell.borrow_mut();
        gs.all_valid_game_moves_for_current_player_into(&mut moves);
        assert!(!moves.is_empty(), "greedy_move_deep_with_score called with no legal moves");
        moves.shuffle(rng);

        let mut capture_end = 0;
        for i in 0..moves.len() {
            if matches!(&moves[i], PossibleMove::ApplyNonCommandTileAction { capturing: Some(_), .. }) {
                moves.swap(i, capture_end);
                capture_end += 1;
            }
        }

        let base_eval_rng = SmallRng::seed_from_u64(0);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0usize;

        for (i, pm) in moves.iter().enumerate() {
            let mv: AiMove = pm.into();
            let undo = mv.to_undo_move().expect("Legal move should be undoable");
            let mut eval_rng = base_eval_rng.clone();
            mv.play(gs, &mut eval_rng);
            let score = -negamax_ab(gs, evaluator, depth - 1, f64::NEG_INFINITY, -best_score, rng, 0);
            gs.undo(undo);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let best_move: AiMove = (&moves[best_idx]).into();
        (best_move, best_score)
    })
}

/// Pick the move that minimizes the opponent's value (= maximizes our value).
///
/// Works with any `GameEvaluator` implementation (NNUE, heuristic, etc.).
/// When the evaluator wraps a sparse `GenericMlp` (1106 or 1147 inputs),
/// automatically uses the incremental L1 accumulator path for faster
/// candidate evaluation.
///
/// Uses make/undo in-place instead of cloning GameState per candidate.
///
/// Panics if the game state has no legal moves.
pub fn greedy_move<E: GameEvaluator + ?Sized>(gs: &mut GameState, evaluator: &E, rng: &mut impl Rng) -> AiMove {
    if let Some((net, include_combined)) = evaluator.as_generic_mlp() {
        return greedy_move_incremental(gs, net, include_combined, rng);
    }
    if evaluator.is_static_heuristic() {
        return greedy_move_heuristic_incremental(gs, rng);
    }

    let owner = gs.current_player_turn();
    GREEDY_CANDIDATES.with(|cell| {
        let mut moves = cell.borrow_mut();
        collect_greedy_candidate_moves(gs, owner, &mut moves);
        assert!(!moves.is_empty(), "greedy_move called with no legal moves");
        moves.shuffle(rng);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        let base_eval_rng = SmallRng::seed_from_u64(0);
        for (i, pm) in moves.iter().enumerate() {
            let mv: AiMove = pm.into();
            let mut eval_rng = base_eval_rng.clone();
            mv.play(gs, &mut eval_rng);
            let is_placement = matches!(pm, PossibleMove::PlaceNewTile(..));
            if is_placement || !gs.is_duke_in_guard(owner) {
                let prediction = evaluator.evaluate(gs);
                let score = -(prediction as f64);
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
            gs.undo(pm.clone());
        }

        (&moves[best_idx]).into()
    })
}

/// Incremental heuristic evaluation: precompute per-piece move counts once,
/// then update only the affected pieces per candidate move.
fn greedy_move_heuristic_incremental(gs: &mut GameState, rng: &mut impl Rng) -> AiMove {
    use duke_rust::common::coordinates::Coordinates;

    let owner = gs.current_player_turn();
    let other = owner.next_player();

    GREEDY_CANDIDATES.with(|cell| {
        let mut moves = cell.borrow_mut();
        collect_greedy_candidate_moves(gs, owner, &mut moves);
        assert!(!moves.is_empty(), "greedy_move_heuristic_incremental called with no legal moves");
        moves.shuffle(rng);

    // Precompute per-piece move counts and base totals.
    let mut piece_counts = [0u16; 36]; // indexed by board position (y*6+x)
    let mut base_own_total = 0i32;
    let mut base_opp_total = 0i32;
    let duke_pos_own = gs.duke_coordinate(owner);
    let duke_pos_opp = gs.duke_coordinate(other);
    let mut base_own_duke = 0i32;
    let mut base_opp_duke = 0i32;

    let mut bits = gs.owner_pieces_bitboard(owner);
    while bits != 0 {
        let idx = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let c = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
        let n = gs.count_legal_moves_for_piece(c) as u16;
        piece_counts[idx] = n;
        base_own_total += n as i32;
        if c == duke_pos_own { base_own_duke = n as i32; }
    }
    let mut bits = gs.owner_pieces_bitboard(other);
    while bits != 0 {
        let idx = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let c = Coordinates { x: (idx % 6) as u8, y: (idx / 6) as u8 };
        let n = gs.count_legal_moves_for_piece(c) as u16;
        piece_counts[idx] = n;
        base_opp_total += n as i32;
        if c == duke_pos_opp { base_opp_duke = n as i32; }
    }

    let base_own_tiles = gs.piece_count(owner) as i32;
    let base_opp_tiles = gs.piece_count(other) as i32;
    let base_discard = gs.discard_bag_for(owner).len() as f64
        - gs.discard_bag_for(other).len() as f64;

    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;

    let base_eval_rng = SmallRng::seed_from_u64(0);
    for (i, pm) in moves.iter().enumerate() {
        let mv: AiMove = pm.into();
        let mut eval_rng = base_eval_rng.clone();
        mv.play(gs, &mut eval_rng);

        let is_placement = matches!(pm, PossibleMove::PlaceNewTile(..));
        if !is_placement && gs.is_duke_in_guard(owner) {
            gs.undo(pm.clone());
            continue;
        }

        // After play, current_player_turn has switched. Evaluate from opponent's POV.
        // We compute the score as if WE (owner) are evaluating.
        // The heuristic formula is: duke_diff + 10*tiles_diff + moves_diff - 15*discard_diff
        // where diff = own - opp, from the current player's perspective.
        // After play, the "current player" is `other`. We want the score from `owner`'s POV,
        // then negate (since greedy_move uses -prediction).

        let score = if is_placement {
            let eval = StaticHeuristicEvaluator::new();
            -(eval.evaluate(gs) as f64)
        } else {
            let (src, dst, capturing) = match pm {
                PossibleMove::ApplyNonCommandTileAction { src, dst, capturing } =>
                    (*src, *dst, capturing.as_ref()),
                _ => unreachable!(),
            };

            let old_src_idx = src.y as usize * 6 + src.x as usize;
            let old_count = piece_counts[old_src_idx] as i32;

            // Determine if this was a Strike (piece stays at src) or Movement (piece moves to dst).
            // After play, check if piece is at dst (Movement) or src (Strike).
            let is_strike = (1u64 << old_src_idx) & gs.owner_pieces_bitboard(owner) != 0;
            let mover_pos = if is_strike { src } else { dst };
            let new_count = gs.count_legal_moves_for_piece(mover_pos) as i32;

            let mut own_total = base_own_total - old_count + new_count;
            let mut opp_total = base_opp_total;
            let own_tiles = base_own_tiles;
            let mut opp_tiles = base_opp_tiles;

            if let Some(_captured) = capturing {
                let cap_idx = dst.y as usize * 6 + dst.x as usize;
                opp_total -= piece_counts[cap_idx] as i32;
                opp_tiles -= 1;
            }

            let own_duke = if src == duke_pos_own { new_count } else { base_own_duke };
            let opp_duke = if capturing.map_or(false, |c| c.tile_type.is_duke()) {
                0
            } else {
                base_opp_duke
            };

            // Compute from `other`'s POV (other is now current_player after play).
            let duke_diff = opp_duke as f64 - own_duke as f64;
            let tiles_diff = 10.0 * (opp_tiles as f64 - own_tiles as f64);
            let moves_diff = opp_total as f64 - own_total as f64;
            let new_discard = gs.discard_bag_for(other).len() as f64
                - gs.discard_bag_for(owner).len() as f64;
            let discard_diff = -15.0 * new_discard;

            let prediction = duke_diff + tiles_diff + moves_diff + discard_diff;
            -prediction
        };

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
        gs.undo(pm.clone());
    }

        (&moves[best_idx]).into()
    })
}

/// Pick the best move using incremental L1 accumulator updates.
///
/// Like `greedy_move`, but exploits the fact that most candidate moves only
/// change 2-4 features in the L1 input.  Builds the base L1 accumulator once
/// from the current position, then for each candidate:
///   1. Make the move in-place (no clone).
///   2. Clone the base accumulator.
///   3. Compute the feature diff (old vs new board/bag/combined features).
///   4. Patch the accumulator with the diff.
///   5. Complete the forward pass (ReLU + remaining layers).
///   6. Undo the move to restore the original state.
///
/// `include_combined` should be `true` for 1147-input models, `false` for 1106.
pub fn greedy_move_incremental(
    gs: &mut GameState,
    net: &GenericMlp,
    include_combined: bool,
    rng: &mut impl Rng,
) -> AiMove {
    let result = GREEDY_CANDIDATES.with(|cell| {
        let mut moves = cell.borrow_mut();
        gs.all_valid_game_moves_for_current_player_into(&mut moves);
        assert!(!moves.is_empty(), "greedy_move_incremental called with no legal moves");
        moves.shuffle(rng);

    let base_acc = L1Accumulator::from_state(net, gs, include_combined);
    let base_board = active_board_features(gs);
    let base_bag = bag_features(gs);
    let base_combined: Option<[f64; NUM_COMBINED_FEATURES]> = if include_combined {
        Some(extract_combined_features(gs))
    } else {
        None
    };

    let base_eval_rng = SmallRng::seed_from_u64(0);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0usize;

    for (i, pm) in moves.iter().enumerate() {
        let mv: AiMove = pm.into();
        let undo = mv.to_undo_move().expect("Legal move should be undoable");
        let mut eval_rng = base_eval_rng.clone();
        mv.play(gs, &mut eval_rng);

        let new_board = active_board_features(gs);
        let new_bag = bag_features(gs);
        let new_combined: Option<[f64; NUM_COMBINED_FEATURES]> = if include_combined {
            Some(extract_combined_features(gs))
        } else {
            None
        };

        let mut acc = base_acc.clone();
        acc.update_features(
            net,
            &base_board,
            &new_board,
            &base_bag,
            &new_bag,
            base_combined.as_ref(),
            new_combined.as_ref(),
        );

        let prediction = acc.forward(net);
        gs.undo(undo);
        let score = -(prediction as f64);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

        (&moves[best_idx]).into()
    });
    result
}

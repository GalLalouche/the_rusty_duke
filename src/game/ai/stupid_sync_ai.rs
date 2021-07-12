use crate::game::board::GameMove;
use crate::game::state::GameState;

pub(crate) fn next_move(gs: &mut GameState) -> () {
    for (pos, _) in gs.get_tiles_for_current_owner() {
        if let Some(a) = gs.get_legal_moves(&pos).first() {
            gs.make_a_move(&GameMove::ApplyNonCommandTileAction { src: pos, dst: a.0 });
            return;
        }
    }
    panic!("Cannot find a simple dumb move... Do'h!")
}
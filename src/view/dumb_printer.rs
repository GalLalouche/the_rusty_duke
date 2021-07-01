use crate::game::state::GameState;
use crate::game::token::{TokenBag, DiscardBag, OwnedToken};

pub fn print(gs: &GameState) -> String {
    fn to_row(row: &Vec<Option<OwnedToken>>) -> String {
        let s = row.iter().map(|o| o.map(|t| t.single_char_token()).or_else(' ')).collect();
        unimplemented!()
    }
    let rows = || {
        gs.board
            .rows()
            .iter()
            .map(|e| to_row(e))
            .collect::<Vec<String>>()
    };
    let board = rows().join("\n");
    unimplemented!()
//     fn summarize(bag: &TokenBag, discard: &DiscardBag) -> String {
//         unimplemented!()
//     }
//     let current_turn = gs.current_player_turn.to_string();
//     let p1_summary = summarize(&gs.player_1_bag, &gs.player_1_discard);
//     let p2_summary = summarize(&gs.player_2_bag, &gs.player_2_discard);
//     [board, p1_summary, p2_summary].join("\n\n")
}
use crate::common::utils::{MkString, Vectors};
use crate::game::state::GameState;
use crate::game::tile::PlacedTile;

pub fn print_board(gs: &GameState) -> String {
    fn to_row(row: &Vec<Option<PlacedTile>>) -> String {
        row.iter()
            .map(|o| o.as_ref().map_or(' ', |t| t.single_char_token()).to_string())
            .collect::<Vec<String>>()
            .mk_string_full("|", "|", "|")
    }
    let rows = || {
        let mut result = gs
            .rows()
            .iter()
            .map(|e| to_row(e))
            .collect::<Vec<String>>();
        let length = result[0].len();
        result.intercalate_full(
            format!("/{}\\", "=".repeat(length - 2)),
            "-".repeat(length),
            format!("\\{}/", "=".repeat(length - 2)),
        );
        result
    };
    let board = rows().join("\n");
    board
}

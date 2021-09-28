use crate::common::board::Board;
use crate::common::utils::{MkString, CloneVectors};
use crate::game::state::GameState;
use crate::game::tile::{CurrentSide, Owner, PlacedTile};

pub fn single_char_print_state(gs: &GameState) -> String {
    single_char_print_board(gs.board())
}

pub fn single_char_print_board(b: &Board<PlacedTile>) -> String {
    fn to_row(row: &Vec<Option<PlacedTile>>) -> String {
        row.iter()
            .map(|o| o.as_ref().map_or(' ', |t| t.single_char_token()).to_string())
            .collect::<Vec<String>>()
            .mk_string_full("|", "|", "|")
    }
    wrap(b, |r| vec![to_row(r)])
}

pub fn double_char_print_state(gs: &GameState) -> String {
    double_char_print_board(gs.board())
}

pub fn double_char_print_board(t: &Board<PlacedTile>) -> String {
    fn name(b: &PlacedTile) -> String {
        b.tile.get_name()[..2].to_owned()
    }
    fn state(b: &PlacedTile) -> String {
        match b.current_side {
            CurrentSide::Initial => "IN".to_owned(),
            CurrentSide::Flipped => "FL".to_owned(),
        }
    }
    fn double_char_token_top_row(t: &PlacedTile) -> String {
        match t.owner {
            Owner::TopPlayer => name(t),
            Owner::BottomPlayer => state(t),
        }
    }
    fn double_char_token_bottom_row(t: &PlacedTile) -> String {
        match t.owner {
            Owner::TopPlayer => state(t),
            Owner::BottomPlayer => name(t),
        }
    }

    fn to_row(row: &Vec<Option<PlacedTile>>) -> Vec<String> {
        let go = |f: fn(&PlacedTile) -> String| {
            row.iter()
                .map(|o| o.as_ref().map_or("  ".to_owned(), f))
                .collect::<Vec<String>>()
                .mk_string_full("|", "|", "|")
        };
        vec![
            go(double_char_token_top_row),
            go(double_char_token_bottom_row),
        ]
    }
    wrap(t, to_row)
}

fn wrap(b: &Board<PlacedTile>, to_row: fn(&Vec<Option<PlacedTile>>) -> Vec<String>) -> String {
    let mut result: Vec<String> = b
        .rows()
        .iter()
        .flat_map(|e| to_row(e))
        .collect::<Vec<String>>();
    let length = result[0].len();
    result.intercalate_every_n(
        format!("/{}\\", "=".repeat(length - 2)),
        "-".repeat(length),
        format!("\\{}/", "=".repeat(length - 2)),
        2,
    );
    result.join("\n")
}
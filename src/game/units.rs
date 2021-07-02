use std::collections::HashMap;

use crate::game::offset::{Centerable, Coordinate, HorizontalOffset, VerticalOffset};
use crate::game::token::{GameToken, TokenAction, TokenSide};

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub fn duke() -> GameToken {
    fn sliders<A: Centerable>(o: A) -> TokenSide {
        let c = Coordinate::centered(o);
        TokenSide::new(
            hashmap![c => TokenAction::Slide, c.flipped() => TokenAction::Slide])
    }
    GameToken::new(
        sliders(HorizontalOffset::Left),
        sliders(VerticalOffset::Top),
        'd',
    )
}

pub fn footman() -> GameToken {
    fn moves(cs: Vec<Coordinate>) -> HashMap<Coordinate, TokenAction> {
        cs.iter().cloned().map(|e| (e, TokenAction::Move)).collect()
    }
    GameToken::new(
        TokenSide::new(moves(vec![
            Coordinate::centered(VerticalOffset::Top),
            Coordinate::centered(VerticalOffset::Bottom),
            Coordinate::centered(HorizontalOffset::Left),
            Coordinate::centered(HorizontalOffset::Right),
        ])),
        TokenSide::new(moves(vec![
            Coordinate { x: HorizontalOffset::Left, y: VerticalOffset::Top },
            Coordinate { x: HorizontalOffset::Right, y: VerticalOffset::Top },
            Coordinate { x: HorizontalOffset::Left, y: VerticalOffset::Bottom },
            Coordinate { x: HorizontalOffset::Right, y: VerticalOffset::Bottom },
            Coordinate::centered(VerticalOffset::FarTop),
        ])),
        'f',
    )
}

use std::collections::HashMap;

use crate::game::offset::{Centerable, Offsets, HorizontalOffset, VerticalOffset};
use crate::game::token::{GameToken, OwnedToken, Owner, TokenAction, TokenSide};

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub fn duke(owner: Owner) -> OwnedToken {
    fn sliders<A: Centerable>(o: A) -> TokenSide {
        let c = Offsets::centered(o);
        TokenSide::new(
            hashmap![c => TokenAction::Slide, c.flipped() => TokenAction::Slide])
    }
    OwnedToken {
        token: GameToken::new(
            sliders(HorizontalOffset::Left),
            sliders(VerticalOffset::Top),
            'd',
        ),
        owner,
    }
}

pub fn footman(owner: Owner) -> OwnedToken {
    fn moves(cs: Vec<Offsets>) -> HashMap<Offsets, TokenAction> {
        cs.iter().cloned().map(|e| (e, TokenAction::Move)).collect()
    }
    OwnedToken {
        token: GameToken::new(
            TokenSide::new(moves(vec![
                Offsets::centered(VerticalOffset::Top),
                Offsets::centered(VerticalOffset::Bottom),
                Offsets::centered(HorizontalOffset::Left),
                Offsets::centered(HorizontalOffset::Right),
            ])),
            TokenSide::new(moves(vec![
                Offsets {
                    x: HorizontalOffset::Left,
                    y: VerticalOffset::Top,
                },
                Offsets { x: HorizontalOffset::Right, y: VerticalOffset::Top },
                Offsets { x: HorizontalOffset::Left, y: VerticalOffset::Bottom },
                Offsets { x: HorizontalOffset::Right, y: VerticalOffset::Bottom },
                Offsets::centered(VerticalOffset::FarTop),
            ])),
            'f',
        ),
        owner,
    }
}

mod test {
    use super::*;

    #[test]
    fn duke_side_1_active_does_not_panic() {
        duke(Owner::Player1).token.side_a.actions();
    }

    #[test]
    fn duke_side_2_active_does_not_panic() {
        duke(Owner::Player1).token.side_b.actions();
    }

    #[test]
    fn footman_side_1_active_does_not_panic() {
        footman(Owner::Player1).token.side_a.actions();
    }

    #[test]
    fn footman_side_2_active_does_not_panic() {
        footman(Owner::Player1).token.side_b.actions();
    }
}

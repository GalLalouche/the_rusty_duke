use std::collections::HashMap;

use crate::game::offset::{Centerable, HorizontalOffset, Offsets, VerticalOffset};
use crate::game::token::{GameToken, OwnedToken, Owner, TokenAction, TokenSide};

macro_rules! hashmap {
    ($($key: expr => $val: expr), *) => {{
         let mut map = ::std::collections::HashMap::new();
         $(map.insert($key, $val);)*
         map
    }}
}

// TODO: all tiles are left/right symmetric (which makes sense, since they are used by both players.
// This should be reflected somehow.
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
            "Duke",
        ),
        owner,
    }
}

fn near_moves(a: TokenAction) -> HashMap<Offsets, TokenAction> {
    vec![
        Offsets::centered(VerticalOffset::Top),
        Offsets::centered(VerticalOffset::Bottom),
        Offsets::centered(HorizontalOffset::Left),
        Offsets::centered(HorizontalOffset::Right),
    ].iter().map(|o| (*o, a)).collect()
}

// pub fn bowman(owner: Owner) -> OwnedToken {
//     OwnedToken {
//         token: GameToken::new(
//             TokenSide::new(hashmap![
//                 Offsets::centered(HorizontalOffset::Left) => TokenAction::Move,
//                 Offsets::centered(HorizontalOffset::Right) => TokenAction::Move,
//                 Offsets::centered(VerticalOffset::Top) => TokenAction::Move,
//
//                 Offsets::centered(HorizontalOffset::FarLeft) => TokenAction::Jump,
//                 Offsets::centered(HorizontalOffset::FarRight) => TokenAction::Jump,
//                 Offsets::centered(VerticalOffset::FarBottom) => TokenAction::Jump,
//             ]),
//             TokenSide::new(hashmap![
//                 Offsets::centered(VerticalOffset::Top) => TokenAction::Move,
//                 Offsets::centered(VerticalOffset::FarTop) => TokenAction::Strike,
//
//                 Offsets::new(HorizontalOffset::Left, VerticalOffset::Top) => TokenAction::Strike,
//                 Offsets::new(HorizontalOffset::Right, VerticalOffset::Top) => TokenAction::Strike,
//                 Offsets::new(HorizontalOffset::FarRight) => TokenAction::Jump,
//                 Offsets::new(VerticalOffset::FarBottom) => TokenAction::Jump,
//             ]),
//             ])),
//             "Bowman",
//         ),
//         owner,
//     }
// }

pub fn footman(owner: Owner) -> OwnedToken {
    fn moves(cs: Vec<Offsets>) -> HashMap<Offsets, TokenAction> {
        cs.iter().cloned().map(|e| (e, TokenAction::Move)).collect()
    }
    OwnedToken {
        token: GameToken::new(
            TokenSide::new(near_moves(TokenAction::Move)),
            TokenSide::new(moves(vec![
                Offsets::new(HorizontalOffset::Left, VerticalOffset::Top),
                Offsets::new(HorizontalOffset::Right, VerticalOffset::Top),
                Offsets::new(HorizontalOffset::Left, VerticalOffset::Bottom),
                Offsets::new(HorizontalOffset::Right, VerticalOffset::Bottom),
                Offsets::centered(VerticalOffset::FarTop),
            ])),
            "Footman",
        ),
        owner,
    }
}

pub fn champion(owner: Owner) -> OwnedToken {
    fn far_moves(a: TokenAction) -> HashMap<Offsets, TokenAction> {
        vec![
            Offsets::centered(VerticalOffset::FarTop),
            Offsets::centered(VerticalOffset::FarBottom),
            Offsets::centered(HorizontalOffset::FarLeft),
            Offsets::centered(HorizontalOffset::FarRight),
        ].iter().map(|o| (*o, a)).collect()
    }
    fn chained(near_action: TokenAction, far_action: TokenAction) -> TokenSide {
        TokenSide::new(near_moves(near_action).into_iter().chain(far_moves(far_action)).collect())
    }
    OwnedToken {
        token: GameToken::new(
            chained(TokenAction::Move, TokenAction::Jump),
            chained(TokenAction::Strike, TokenAction::Jump),
            "Champion",
        ),
        owner,
    }
}

pub fn wizard(owner: Owner) -> OwnedToken {
    OwnedToken {
        token: GameToken::new(
            TokenSide::new(
                vec![
                    Offsets::new(HorizontalOffset::Left, VerticalOffset::Top),
                    Offsets::new(HorizontalOffset::Left, VerticalOffset::Center),
                    Offsets::new(HorizontalOffset::Left, VerticalOffset::Bottom),
                    Offsets::new(HorizontalOffset::Center, VerticalOffset::Top),
                    Offsets::new(HorizontalOffset::Center, VerticalOffset::Bottom),
                    Offsets::new(HorizontalOffset::Right, VerticalOffset::Top),
                    Offsets::new(HorizontalOffset::Right, VerticalOffset::Center),
                    Offsets::new(HorizontalOffset::Right, VerticalOffset::Bottom),
                ].iter().map(|o| (*o, TokenAction::Move)).collect()
            ),
            TokenSide::new(
                vec![
                    Offsets::new(HorizontalOffset::FarLeft, VerticalOffset::FarTop),
                    Offsets::new(HorizontalOffset::FarLeft, VerticalOffset::Center),
                    Offsets::new(HorizontalOffset::FarLeft, VerticalOffset::FarBottom),
                    Offsets::new(HorizontalOffset::Center, VerticalOffset::FarBottom),
                    Offsets::new(HorizontalOffset::FarRight, VerticalOffset::FarTop),
                    Offsets::new(HorizontalOffset::FarRight, VerticalOffset::Center),
                    Offsets::new(HorizontalOffset::FarRight, VerticalOffset::FarBottom),
                ].iter().map(|o| (*o, TokenAction::Jump)).collect()
            ),
            "Wizard",
        ),
        owner,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use paste::paste;

    macro_rules! no_panics {
        ($($ctor: ident),+) => {
            $(paste! {
                #[test]
                fn [<$ctor _side_a_active_does_not_panic>]() {
                    $ctor(Owner::Player1).token.side_a.actions();
                }
                #[test]
                fn [<$ctor _side_b_active_does_not_panic>]() {
                    $ctor(Owner::Player1).token.side_b.actions();
                }
            })+
        }
    }

    no_panics!(duke, footman, champion, wizard);
}

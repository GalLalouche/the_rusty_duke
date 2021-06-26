use crate::token;

macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

fn duke() -> token::GameToken {
    fn sliders<A: token::Centerable>(o: A) -> token::Side {
        let c = token::Coordinate::centered(o);
        token::Side::new(
            hashmap![c => token::TokenAction::Slide, c.flipped() => token::TokenAction::Slide])
    }
    token::GameToken::new(
        sliders(token::HorizontalOffset::Left),
        sliders(token::VerticalOffset::Top),
    )
}

fn footman() -> token::GameToken {
    token::GameToken::new(
        token::Side::new(hashmap![
            token::Coordinate::centered(token::VerticalOffset::Top) => token::TokenAction::Move,
            token::Coordinate::centered(token::VerticalOffset::Bottom) => token::TokenAction::Move,
            token::Coordinate::centered(token::HorizontalOffset::Left) => token::TokenAction::Move,
            token::Coordinate::centered(token::HorizontalOffset::Right) => token::TokenAction::Move,
        ]),
        token::Side::new(hashmap![
            token::Coordinate{ x: token::VerticalOffset::Top, token::) => token::TokenAction::Move,
            token::Coordinate::centered(token::VerticalOffset::Bottom) => token::TokenAction::Move,
            token::Coordinate::centered(token::HorizontalOffset::Left) => token::TokenAction::Move,
            token::Coordinate::centered(token::HorizontalOffset::Right) => token::TokenAction::Move,
        ]),
    )
}

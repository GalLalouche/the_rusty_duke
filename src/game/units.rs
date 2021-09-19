use crate::game::offset::{FourWaySymmetric, HorizontalSymmetricOffset, VerticalOffset};
use crate::game::tile::{Owner, PlacedTile, Tile, TileAction, TileSide};

const DUKE_NAME: &str = "Duke";

pub fn is_duke(t: &Tile) -> bool {
    t.get_name() == DUKE_NAME
}

pub fn duke() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Slide)
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Slide),
            (&VerticalOffset::Bottom, TileAction::Slide),
        ]),
        DUKE_NAME,
    )
}

pub fn bowman() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&VerticalOffset::FarBottom, TileAction::Jump),
            (&HorizontalSymmetricOffset::Near, TileAction::Move),
            (&HorizontalSymmetricOffset::Far, TileAction::Jump),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&VerticalOffset::FarTop, TileAction::Strike),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Top), TileAction::Strike),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Move),
        ]),
        "Bowman",
    )
}

pub fn footman() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Move)
        ]),
        TileSide::new(vec![
            (&FourWaySymmetric::NearDiagonal, TileAction::Move),
            (&VerticalOffset::FarTop, TileAction::Move),
        ]),
        "Footman",
    )
}

pub fn dragoon() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Move),
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarTop), TileAction::Strike),
            (&VerticalOffset::FarTop, TileAction::Strike),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&VerticalOffset::FarTop, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarTop), TileAction::Jump),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Slide),
        ]),
        "Dragoon",
    )
}

pub fn assassin() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarBottom), TileAction::JumpSlide),
            (&VerticalOffset::FarTop, TileAction::JumpSlide),
        ]),
        TileSide::new(vec![
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarTop), TileAction::JumpSlide),
            (&VerticalOffset::FarBottom, TileAction::JumpSlide),
        ]),
        "Assassin",
    )
}

pub fn champion() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Move),
            (&FourWaySymmetric::FarStraight, TileAction::Jump),
        ]),
        TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Strike),
            (&FourWaySymmetric::FarStraight, TileAction::Jump),
        ]),
        "Champion",
    )
}

pub fn general() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&VerticalOffset::Bottom, TileAction::Move),
            (&HorizontalSymmetricOffset::Far, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarTop), TileAction::Jump),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&HorizontalSymmetricOffset::Near, TileAction::Move),
            (&HorizontalSymmetricOffset::Far, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarTop), TileAction::Jump),
            (&HorizontalSymmetricOffset::Near, TileAction::Command),
            (&VerticalOffset::Bottom, TileAction::Command),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Command),
        ]),
        "General",
    )
}

pub fn marshall() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarTop), TileAction::Jump),
            (&HorizontalSymmetricOffset::Near, TileAction::Slide),
            (&VerticalOffset::FarBottom, TileAction::Jump),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&HorizontalSymmetricOffset::Near, TileAction::Move),
            (&HorizontalSymmetricOffset::Far, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Top), TileAction::Move),
            (&VerticalOffset::Top, TileAction::Command),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Top), TileAction::Command),
        ]),
        "Marshall",
    )
}

pub fn priest() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&FourWaySymmetric::NearDiagonal, TileAction::Slide),
        ]),
        TileSide::new(vec![
            (&FourWaySymmetric::NearDiagonal, TileAction::Move),
            (&FourWaySymmetric::FarDiagonal, TileAction::Jump),
        ]),
        "Priest",
    )
}

pub fn longbowman() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&VerticalOffset::Bottom, TileAction::Unit),
            (&VerticalOffset::Center, TileAction::Move),
            (&VerticalOffset::FarBottom, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Move),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Bottom, TileAction::Unit),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarBottom), TileAction::Move),
            (&VerticalOffset::Top, TileAction::Strike),
            (&VerticalOffset::FarTop, TileAction::Strike),
        ]),
        "Longbowman",
    )
}

pub fn knight() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Move),
            (&VerticalOffset::Bottom, TileAction::Move),
            (&VerticalOffset::FarBottom, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarTop), TileAction::Jump),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Slide),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Bottom), TileAction::Move),
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarBottom), TileAction::Move),
        ]),
        "Knight",
    )
}

pub fn pikeman() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::Top), TileAction::Move),
            (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarTop), TileAction::Move),
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Move),
            (&VerticalOffset::Bottom, TileAction::Move),
            (&VerticalOffset::FarBottom, TileAction::Move),
            (&(HorizontalSymmetricOffset::Near, VerticalOffset::FarTop), TileAction::Strike),
        ]),
        "Pikeman",
    )
}

pub fn wizard() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&FourWaySymmetric::NearStraight, TileAction::Move),
            (&FourWaySymmetric::NearDiagonal, TileAction::Move),
        ]),
        TileSide::new(vec![
            (&FourWaySymmetric::FarStraight, TileAction::Jump),
            (&FourWaySymmetric::FarDiagonal, TileAction::Jump),
        ]),
        "Wizard",
    )
}

#[cfg(test)]
mod test {
    use paste::paste;

    use super::*;

    macro_rules! no_panics {
        ($($ctor: ident),+ $(,)?) => {
            $(paste! {
                #[test]
                fn [<$ctor _side_a_active_does_not_panic>]() {
                    $ctor().get_side_a().actions();
                }
                #[test]
                fn [<$ctor _side_b_active_does_not_panic>]() {
                    $ctor().get_side_b().actions();
                }
            })+
        }
    }

    no_panics!(
        duke,
        bowman,
        dragoon,
        assassin,
        champion,
        footman,

        general,
        marshall,
        priest,
        longbowman,
        knight,
        pikeman,
        wizard,

        //TODO add box units, like Light Horse.
    );
}

pub fn place_tile<U>(o: Owner, ctor: U) -> PlacedTile where U: Fn() -> Tile {
    PlacedTile::new(o, ctor())
}
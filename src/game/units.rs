use crate::game::offset::{FourWaySymmetric, HorizontalSymmetricOffset, VerticalOffset};
use crate::game::tile::{Owner, PlacedTile, Tile, TileType};
use crate::game::tile_side::{TileAction, TileSide};

pub fn duke() -> Tile {
    Tile::new(
        TileSide::new(vec![
            (&HorizontalSymmetricOffset::Near, TileAction::Slide)
        ]),
        TileSide::new(vec![
            (&VerticalOffset::Top, TileAction::Slide),
            (&VerticalOffset::Bottom, TileAction::Slide),
        ]),
        TileType::Duke,
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
        TileType::Bowman,
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
        TileType::Footman,
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
        TileType::Dragoon,
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
        TileType::Assassin,
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
        TileType::Champion,
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
        TileType::General,
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
        TileType::Marshall,
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
        TileType::Priest,
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
        TileType::Longbowman,
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
        TileType::Knight,
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
        TileType::Pikeman,
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
        TileType::Wizard,
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

    #[test]
    fn tile_from_type_does_not_panic_for_any_variant() {
        use std::convert::TryFrom;
        use strum::EnumCount;
        for i in 0..TileType::COUNT {
            let tt = TileType::try_from(i as u8).unwrap();
            let tile = tile_from_type(tt);
            assert_eq!(
                tile.tile_type(),
                tt,
                "tile_from_type({:?}) returned tile with wrong type {:?}",
                tt,
                tile.tile_type(),
            );
        }
    }

    #[test]
    fn tile_from_type_returns_correct_name_for_each_variant() {
        let expected: Vec<(&str, TileType)> = vec![
            ("Duke", TileType::Duke),
            ("Footman", TileType::Footman),
            ("Pikeman", TileType::Pikeman),
            ("Knight", TileType::Knight),
            ("Champion", TileType::Champion),
            ("Dragoon", TileType::Dragoon),
            ("Wizard", TileType::Wizard),
            ("General", TileType::General),
            ("Marshall", TileType::Marshall),
            ("Assassin", TileType::Assassin),
            ("Priest", TileType::Priest),
            ("Bowman", TileType::Bowman),
            ("Longbowman", TileType::Longbowman),
        ];
        for (name, tt) in expected {
            let tile = tile_from_type(tt);
            assert_eq!(
                tile.get_name(),
                name,
                "tile_from_type({:?}) has wrong name",
                tt,
            );
        }
    }
}

/// Construct a Tile from its TileType.
pub fn tile_from_type(tt: TileType) -> Tile {
    match tt {
        TileType::Duke => duke(),
        TileType::Footman => footman(),
        TileType::Pikeman => pikeman(),
        TileType::Knight => knight(),
        TileType::Champion => champion(),
        TileType::Dragoon => dragoon(),
        TileType::Wizard => wizard(),
        TileType::General => general(),
        TileType::Marshall => marshall(),
        TileType::Assassin => assassin(),
        TileType::Priest => priest(),
        TileType::Bowman => bowman(),
        TileType::Longbowman => longbowman(),
    }
}

pub fn place_tile<U>(o: Owner, ctor: U) -> PlacedTile where U: Fn() -> Tile {
    PlacedTile::new(o, ctor())
}

pub fn place_tile_flipped<U>(o: Owner, ctor: U) -> PlacedTile where U: Fn() -> Tile {
    let mut result = PlacedTile::new(o, ctor());
    result.flip();
    result
}

use crate::game::offset::{FourWaySymmetric, HorizontalSymmetricOffset, VerticalOffset};
use crate::game::tile::{Tile, OwnedTile, Owner, TileAction, TileSide};

pub fn duke(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&HorizontalSymmetricOffset::Near, TileAction::Slide)
            ]),
            TileSide::new(vec![
                (&VerticalOffset::Top, TileAction::Slide),
                (&VerticalOffset::Bottom, TileAction::Slide),
            ]),
            "Duke",
        ),
        owner,
    }
}

pub fn bowman(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn footman(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&FourWaySymmetric::NearLinear, TileAction::Move)
            ]),
            TileSide::new(vec![
                (&FourWaySymmetric::NearDiagonal, TileAction::Move),
                (&VerticalOffset::FarTop, TileAction::Move),
            ]),
            "Footman",
        ),
        owner,
    }
}

pub fn dragoon(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn assassin(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarBottom), TileAction::JumpSlide),
                (&VerticalOffset::FarTop, TileAction::JumpSlide),
            ]),
            TileSide::new(vec![
                (&(HorizontalSymmetricOffset::Far, VerticalOffset::FarTop), TileAction::JumpSlide),
                (&VerticalOffset::FarBottom, TileAction::JumpSlide),
            ]),
            "Assassin",
        ),
        owner,
    }
}

pub fn champion(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&FourWaySymmetric::NearLinear, TileAction::Move),
                (&FourWaySymmetric::FarLinear, TileAction::Jump),
            ]),
            TileSide::new(vec![
                (&FourWaySymmetric::NearLinear, TileAction::Strike),
                (&FourWaySymmetric::FarLinear, TileAction::Jump),
            ]),
            "Champion",
        ),
        owner,
    }
}

pub fn general(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn marshall(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn priest(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&FourWaySymmetric::NearDiagonal, TileAction::Slide),
            ]),
            TileSide::new(vec![
                (&FourWaySymmetric::NearDiagonal, TileAction::Move),
                (&FourWaySymmetric::FarDiagonal, TileAction::Jump),
            ]),
            "Priest",
        ),
        owner,
    }
}

pub fn knight(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn pikeman(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
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
        ),
        owner,
    }
}

pub fn wizard(owner: Owner) -> OwnedTile {
    OwnedTile {
        tile: Tile::new(
            TileSide::new(vec![
                (&FourWaySymmetric::NearLinear, TileAction::Move),
                (&FourWaySymmetric::NearDiagonal, TileAction::Move),
            ]),
            TileSide::new(vec![
                (&FourWaySymmetric::FarLinear, TileAction::Jump),
                (&FourWaySymmetric::FarDiagonal, TileAction::Jump),
            ]),
            "Wizard",
        ),
        owner,
    }
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
                    $ctor(Owner::Player1).tile.side_a.actions();
                }
                #[test]
                fn [<$ctor _side_b_active_does_not_panic>]() {
                    $ctor(Owner::Player1).tile.side_b.actions();
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
        // TODO: Longbownman
        knight,
        pikeman,
        wizard,
    );
}

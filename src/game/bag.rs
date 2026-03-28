use rand::Rng;

use crate::game::tile::TileType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBag {
    bag: Vec<TileType>,
}

impl TileBag {
    #[cfg(test)]
    pub fn empty() -> TileBag {
        TileBag { bag: Vec::new() }
    }
    pub fn new(bag: Vec<TileType>) -> TileBag {
        TileBag { bag }
    }

    pub fn pull<R: Rng>(&mut self, rng: &mut R) -> Option<TileType> {
        if self.bag.is_empty() {
            None
        } else {
            let index = rng.gen_range(0..self.bag.len());
            Some(self.bag.remove(index))
        }
    }

    pub fn remaining(&self) -> &Vec<TileType> {
        &self.bag
    }
    pub fn is_empty(&self) -> bool {
        self.remaining().len() == 0
    }
    pub fn non_empty(&self) -> bool {
        !self.is_empty()
    }

    // For undoing
    pub fn push(&mut self, t: TileType) -> () {
        self.bag.push(t);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscardBag {
    bag: Vec<TileType>,
}

impl DiscardBag {
    pub fn empty() -> DiscardBag {
        DiscardBag { bag: Vec::new() }
    }
    pub fn from_tiles(bag: Vec<TileType>) -> DiscardBag {
        DiscardBag { bag }
    }

    pub fn add(&mut self, t: TileType) -> () {
        self.bag.push(t);
    }

    pub fn existing(&self) -> &Vec<TileType> {
        &self.bag
    }

    pub fn len(&self) -> usize { self.bag.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::utils::test_rng;

    // ── TileBag tests ─────────────────────────────────────────────────
    #[test]
    fn pull_removes_a_tile_from_bag() {
        let mut bag = TileBag::new(vec![TileType::Footman, TileType::Knight]);
        let mut rng = test_rng();
        let pulled = bag.pull(&mut rng);
        assert!(pulled.is_some());
        assert_eq!(bag.remaining().len(), 1);
    }

    #[test]
    fn pull_returns_none_from_empty_bag() {
        let mut bag = TileBag::empty();
        let mut rng = test_rng();
        let pulled = bag.pull(&mut rng);
        assert!(pulled.is_none());
    }

    #[test]
    fn pull_all_tiles_empties_bag() {
        let mut bag = TileBag::new(vec![TileType::Footman, TileType::Knight, TileType::Pikeman]);
        let mut rng = test_rng();
        let mut pulled = Vec::new();
        while let Some(tile) = bag.pull(&mut rng) {
            pulled.push(tile);
        }
        assert_eq!(pulled.len(), 3);
        assert!(bag.is_empty());
        assert!(bag.pull(&mut rng).is_none());
    }

    #[test]
    fn is_empty_returns_true_for_empty_bag() {
        let bag = TileBag::empty();
        assert!(bag.is_empty());
    }

    #[test]
    fn is_empty_returns_false_for_non_empty_bag() {
        let bag = TileBag::new(vec![TileType::Footman]);
        assert!(!bag.is_empty());
    }

    #[test]
    fn non_empty_returns_true_for_non_empty_bag() {
        let bag = TileBag::new(vec![TileType::Footman]);
        assert!(bag.non_empty());
    }

    #[test]
    fn non_empty_returns_false_for_empty_bag() {
        let bag = TileBag::empty();
        assert!(!bag.non_empty());
    }

    #[test]
    fn push_adds_tile_back_to_bag() {
        let mut bag = TileBag::new(vec![TileType::Footman]);
        bag.push(TileType::Knight);
        assert_eq!(bag.remaining().len(), 2);
    }

    #[test]
    fn remaining_returns_all_tiles() {
        let tiles = vec![TileType::Footman, TileType::Knight, TileType::Pikeman];
        let bag = TileBag::new(tiles.clone());
        assert_eq!(*bag.remaining(), tiles);
    }

    // ── DiscardBag tests ──────────────────────────────────────────────
    #[test]
    fn discard_bag_empty_starts_empty() {
        let bag = DiscardBag::empty();
        assert_eq!(bag.len(), 0);
        assert!(bag.existing().is_empty());
    }

    #[test]
    fn discard_bag_from_tiles_has_initial_contents() {
        let tiles = vec![TileType::Footman, TileType::Knight];
        let bag = DiscardBag::from_tiles(tiles.clone());
        assert_eq!(bag.len(), 2);
        assert_eq!(*bag.existing(), tiles);
    }

    #[test]
    fn discard_bag_add_increases_count() {
        let mut bag = DiscardBag::empty();
        bag.add(TileType::Footman);
        assert_eq!(bag.len(), 1);
        bag.add(TileType::Knight);
        assert_eq!(bag.len(), 2);
    }

    #[test]
    fn discard_bag_add_preserves_order() {
        let mut bag = DiscardBag::empty();
        bag.add(TileType::Footman);
        bag.add(TileType::Knight);
        bag.add(TileType::Pikeman);
        assert_eq!(
            *bag.existing(),
            vec![TileType::Footman, TileType::Knight, TileType::Pikeman],
        );
    }
}

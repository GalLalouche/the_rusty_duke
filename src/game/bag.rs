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
            // swap_remove is O(1) vs O(n) shift; bag order doesn't matter since draws are random.
            Some(self.bag.swap_remove(index))
        }
    }

    pub fn remaining(&self) -> &Vec<TileType> {
        &self.bag
    }
    pub fn is_empty(&self) -> bool {
        self.bag.is_empty()
    }
    pub fn non_empty(&self) -> bool {
        !self.is_empty()
    }

    /// Remove one instance of a specific tile type from the bag.
    /// Returns `true` if the tile was found and removed, `false` otherwise.
    ///
    /// Uses `swap_remove` for O(1) removal (bag order doesn't matter since
    /// draws are random), consistent with [`pull`].
    pub fn remove_specific(&mut self, tile: TileType) -> bool {
        if let Some(idx) = self.bag.iter().position(|t| *t == tile) {
            self.bag.swap_remove(idx);
            true
        } else {
            false
        }
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

    /// Remove one instance of `t` from the discard pile (used when undoing a capture).
    /// Panics if `t` is not present.
    pub fn remove(&mut self, t: TileType) {
        let idx = self.bag.iter().rposition(|x| *x == t)
            .expect("Tried to remove a tile from discard that isn't there");
        self.bag.swap_remove(idx);
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

    // ── remove_specific tests ────────────────────────────────────────

    #[test]
    fn remove_specific_removes_tile_and_returns_true() {
        let mut bag = TileBag::new(vec![TileType::Footman, TileType::Knight, TileType::Pikeman]);
        assert!(bag.remove_specific(TileType::Knight));
        assert_eq!(bag.remaining().len(), 2);
        assert!(!bag.remaining().contains(&TileType::Knight));
    }

    #[test]
    fn remove_specific_returns_false_if_not_found() {
        let mut bag = TileBag::new(vec![TileType::Footman, TileType::Knight]);
        assert!(!bag.remove_specific(TileType::Pikeman));
        assert_eq!(bag.remaining().len(), 2);
    }

    #[test]
    fn remove_specific_removes_only_one_duplicate() {
        let mut bag = TileBag::new(vec![TileType::Pikeman, TileType::Pikeman, TileType::Pikeman]);
        assert!(bag.remove_specific(TileType::Pikeman));
        assert_eq!(bag.remaining().len(), 2);
        assert!(bag.remaining().iter().all(|t| *t == TileType::Pikeman));
    }

    #[test]
    fn remove_specific_from_empty_bag_returns_false() {
        let mut bag = TileBag::empty();
        assert!(!bag.remove_specific(TileType::Footman));
    }

    // ── DiscardBag::remove tests ─────────────────────────────────────

    #[test]
    fn discard_bag_remove_existing_tile() {
        let mut bag = DiscardBag::from_tiles(vec![TileType::Footman, TileType::Knight]);
        bag.remove(TileType::Footman);
        assert_eq!(bag.len(), 1);
        assert_eq!(bag.existing(), &vec![TileType::Knight]);
    }

    #[test]
    #[should_panic(expected = "Tried to remove a tile from discard")]
    fn discard_bag_remove_nonexistent_panics() {
        let mut bag = DiscardBag::from_tiles(vec![TileType::Footman]);
        bag.remove(TileType::Knight);
    }

    /// Regression test: `remove_specific` previously used `Vec::remove` (O(n))
    /// which shifted remaining elements. Now uses `swap_remove` (O(1)) consistent
    /// with `pull`. Verify that the removed tile is gone and the remaining tile
    /// count is correct, regardless of internal ordering.
    #[test]
    fn remove_specific_swap_remove_preserves_other_tiles() {
        // Set up bag with distinct tile types where removal of the first would
        // shift elements under the old `Vec::remove`, but swap_remove moves the
        // last element to the removed index.
        let mut bag = TileBag::new(vec![
            TileType::Footman, TileType::Knight, TileType::Pikeman, TileType::Champion,
        ]);
        assert!(bag.remove_specific(TileType::Footman));
        assert_eq!(bag.remaining().len(), 3);
        // Footman should be gone
        assert!(!bag.remaining().contains(&TileType::Footman));
        // All other tiles should still be present (as a multiset)
        let mut remaining_sorted: Vec<TileType> = bag.remaining().clone();
        remaining_sorted.sort_by_key(|t| t.index());
        let mut expected = vec![TileType::Knight, TileType::Pikeman, TileType::Champion];
        expected.sort_by_key(|t| t.index());
        assert_eq!(remaining_sorted, expected,
            "remove_specific should preserve all other tiles");
    }
}

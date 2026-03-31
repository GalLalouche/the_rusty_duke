use rand::Rng;

use crate::game::tile::TileType;

/// Maximum number of tiles in a single player's tile bag.
/// The standard game has 12 tile types (excluding Duke which starts on board).
const MAX_BAG_SIZE: usize = 12;

/// Maximum number of tiles in a single player's discard pile.
/// A player starts with Duke + 2 footmen on board + up to 12 in bag = 15 total.
/// All except the duke can be captured (duke capture ends the game), so max 14.
/// Use 16 for safety.
const MAX_DISCARD_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBag {
    tiles: [TileType; MAX_BAG_SIZE],
    len: u8,
}

impl TileBag {
    #[cfg(test)]
    pub fn empty() -> TileBag {
        TileBag {
            tiles: [TileType::Footman; MAX_BAG_SIZE], // placeholder values, len=0 means unused
            len: 0,
        }
    }
    pub fn new(bag: Vec<TileType>) -> TileBag {
        debug_assert!(bag.len() <= MAX_BAG_SIZE,
            "TileBag::new called with {} tiles, max is {}", bag.len(), MAX_BAG_SIZE);
        let mut tiles = [TileType::Footman; MAX_BAG_SIZE];
        for (i, &t) in bag.iter().enumerate() {
            tiles[i] = t;
        }
        TileBag { tiles, len: bag.len() as u8 }
    }

    #[inline]
    pub fn pull<R: Rng>(&mut self, rng: &mut R) -> Option<TileType> {
        if self.len == 0 {
            None
        } else {
            let index = rng.gen_range(0..self.len as usize);
            let tile = self.tiles[index];
            // swap_remove: move last element to the removed index
            self.len -= 1;
            self.tiles[index] = self.tiles[self.len as usize];
            Some(tile)
        }
    }

    #[inline]
    pub fn remaining(&self) -> &[TileType] {
        &self.tiles[..self.len as usize]
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn non_empty(&self) -> bool {
        self.len > 0
    }

    /// Remove one instance of a specific tile type from the bag.
    /// Returns `true` if the tile was found and removed, `false` otherwise.
    ///
    /// Uses swap_remove for O(1) removal (bag order doesn't matter since
    /// draws are random), consistent with [`pull`].
    pub fn remove_specific(&mut self, tile: TileType) -> bool {
        if let Some(idx) = self.remaining().iter().position(|t| *t == tile) {
            self.len -= 1;
            self.tiles[idx] = self.tiles[self.len as usize];
            true
        } else {
            false
        }
    }

    // For undoing
    #[inline]
    pub fn push(&mut self, t: TileType) {
        debug_assert!((self.len as usize) < MAX_BAG_SIZE,
            "TileBag::push overflow: already at max capacity {}", MAX_BAG_SIZE);
        self.tiles[self.len as usize] = t;
        self.len += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscardBag {
    tiles: [TileType; MAX_DISCARD_SIZE],
    len: u8,
}

impl DiscardBag {
    pub fn empty() -> DiscardBag {
        DiscardBag {
            tiles: [TileType::Footman; MAX_DISCARD_SIZE],
            len: 0,
        }
    }
    pub fn from_tiles(bag: Vec<TileType>) -> DiscardBag {
        debug_assert!(bag.len() <= MAX_DISCARD_SIZE,
            "DiscardBag::from_tiles called with {} tiles, max is {}", bag.len(), MAX_DISCARD_SIZE);
        let mut tiles = [TileType::Footman; MAX_DISCARD_SIZE];
        for (i, &t) in bag.iter().enumerate() {
            tiles[i] = t;
        }
        DiscardBag { tiles, len: bag.len() as u8 }
    }

    #[inline]
    pub fn add(&mut self, t: TileType) {
        debug_assert!((self.len as usize) < MAX_DISCARD_SIZE,
            "DiscardBag::add overflow: already at max capacity {}", MAX_DISCARD_SIZE);
        self.tiles[self.len as usize] = t;
        self.len += 1;
    }

    /// Remove one instance of `t` from the discard pile (used when undoing a capture).
    /// Panics if `t` is not present.
    #[inline]
    pub fn remove(&mut self, t: TileType) {
        let idx = self.existing().iter().rposition(|x| *x == t)
            .expect("Tried to remove a tile from discard that isn't there");
        self.len -= 1;
        self.tiles[idx] = self.tiles[self.len as usize];
    }

    pub fn existing(&self) -> &[TileType] {
        &self.tiles[..self.len as usize]
    }

    pub fn len(&self) -> usize { self.len as usize }
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
        assert_eq!(bag.remaining(), &tiles[..]);
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
        assert_eq!(bag.existing(), &tiles[..]);
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
            bag.existing(),
            &[TileType::Footman, TileType::Knight, TileType::Pikeman],
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
        // After swap_remove, the remaining tile is Knight
        assert_eq!(bag.existing(), &[TileType::Knight]);
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
        let mut bag = TileBag::new(vec![
            TileType::Footman, TileType::Knight, TileType::Pikeman, TileType::Champion,
        ]);
        assert!(bag.remove_specific(TileType::Footman));
        assert_eq!(bag.remaining().len(), 3);
        // Footman should be gone
        assert!(!bag.remaining().contains(&TileType::Footman));
        // All other tiles should still be present (as a multiset)
        let mut remaining_sorted: Vec<TileType> = bag.remaining().to_vec();
        remaining_sorted.sort_by_key(|t| t.index());
        let mut expected = vec![TileType::Knight, TileType::Pikeman, TileType::Champion];
        expected.sort_by_key(|t| t.index());
        assert_eq!(remaining_sorted, expected,
            "remove_specific should preserve all other tiles");
    }
}

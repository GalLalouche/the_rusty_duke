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

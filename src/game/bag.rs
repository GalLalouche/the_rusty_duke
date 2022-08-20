use rand::Rng;

use crate::game::tile::TileRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBag {
    bag: Vec<TileRef>,
}

impl TileBag {
    #[cfg(test)]
    pub fn empty() -> TileBag {
        TileBag { bag: Vec::new() }
    }
    pub fn new(bag: Vec<TileRef>) -> TileBag {
        TileBag { bag }
    }

    pub fn pull<R: Rng>(&mut self, rng: &mut R) -> Option<TileRef> {
        if self.bag.is_empty() {
            None
        } else {
            let index = rng.gen_range(0..self.bag.len());
            Some(self.bag.remove(index))
        }
    }

    pub fn remaining(&self) -> &Vec<TileRef> {
        &self.bag
    }
    pub fn is_empty(&self) -> bool {
        self.remaining().len() == 0
    }
    pub fn non_empty(&self) -> bool {
        !self.is_empty()
    }

    // For undoing
    pub fn push(&mut self, t: TileRef) -> () {
        self.bag.push(t);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscardBag {
    bag: Vec<TileRef>,
}

impl DiscardBag {
    pub fn empty() -> DiscardBag {
        DiscardBag { bag: Vec::new() }
    }

    pub fn add(&mut self, t: TileRef) -> () {
        self.bag.push(t);
    }

    pub fn existing(&self) -> &Vec<TileRef> {
        &self.bag
    }

    pub fn len(&self) -> usize { self.bag.len() }
}

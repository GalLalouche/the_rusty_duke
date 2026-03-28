use rand::Rng;

pub struct Percentage(f64);
impl Percentage {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0);
        Percentage(p)
    }

    pub fn roll<R: Rng>(&self, rng: &mut R) -> bool {
        rng.gen_range(0.0..1.0) <= self.0
    }
}
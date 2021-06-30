pub(crate) trait Distance {
    fn distance_to(&self, other: &Self) -> Self;
}

impl Distance for usize {
    fn distance_to(&self, other: &Self) -> Self {
        if self < other {
            other - self
        } else {
            self - other
        }
    }
}

#[macro_export]
macro_rules! assert_not {
    ($b: expr) => {assert!($b)};
    ($b: expr, $msg: expr) => {assert!($b, $msg)};
}
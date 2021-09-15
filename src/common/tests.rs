#[macro_export] macro_rules! assert_empty {
    ($e: expr) => {assert!($e.is_empty())}
}
#[macro_export] macro_rules! assert_none {
    ($e: expr) => {assert_eq!(None, $e)}
}
#[macro_export] macro_rules! assert_some {
    ($s: expr, $e: expr $(,)?) => {assert_eq!(Some($s), $e)}
}
#[macro_export] macro_rules! assert_eq_set {
    ($expected: expr, $actual: expr $(,)?) => {{
        use std::collections::HashSet;
        use std::iter::FromIterator;
        let hs: HashSet<_> = HashSet::from_iter($expected);
        assert_eq!(hs, HashSet::from_iter($actual));
    }}
}

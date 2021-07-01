#[macro_export] macro_rules! assert_none {
    ($e: expr) => {
        assert_eq!(None, $e)
    }
}
#[macro_export]  macro_rules! assert_some {
    ($s: expr, $e: expr) => {
        assert_eq!(Some($s), $e)
    }
}

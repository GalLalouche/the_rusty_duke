use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

#[macro_export] macro_rules! assert_empty {
    ($e: expr) => {
        let x = $e;
        assert!(x.is_empty(), "expected empty, got {:?}", &x)
    }
}
#[macro_export] macro_rules! assert_none {
    ($e: expr) => {assert_eq!(None, $e)}
}
#[macro_export] macro_rules! assert_some {
    ($s: expr, $e: expr $(,)?) => {assert_eq!(Some($s), $e)}
}
pub fn eq_set_msg<A>(expected: HashSet<A>, actual: HashSet<A>) -> Option<String> where A: Eq + Hash + Debug + Clone {
    if expected == actual {
        return None;
    }
    let mut result = "expected set is different from actual set...\n".to_owned();
    let mut add_missing = |set1: &HashSet<A>, set2: &HashSet<A>, str1| -> () {
        let missing: Vec<A> = set1.clone().iter().filter(|a| !set2.contains(a)).cloned().collect();
        if missing.is_empty() {
            return;
        }
        result.push_str(format!("Actual {} {} items:\n", str1, missing.len()).as_str());
        result.push_str(format!("{:?}\n", missing).as_str());
    };
    add_missing(&actual, &expected, "has extra");
    add_missing(&expected, &actual, "is missing");
    Some(result)
}
#[macro_export] macro_rules! assert_eq_set {
    ($expected: expr, $actual: expr $(,)?) => {{
        use std::collections::HashSet;
        use std::iter::FromIterator;
        if let Some(s) = crate::common::tests::eq_set_msg(
                HashSet::from_iter($expected), HashSet::from_iter($actual)) {
            assert!(false, "{}", s);
        }
    }}
}

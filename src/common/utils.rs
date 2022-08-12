use std::cmp::{min, Ordering};
use std::fmt::Debug;

#[macro_export]
macro_rules! assert_not {
    ($cond: expr) => {assert!(!$cond)};
    ($cond: expr, $($arg: tt)+) => {assert! (!$cond, $($arg)*)};
}

pub trait Distance {
    fn distance_to(&self, other: Self) -> Self;
}

impl Distance for usize {
    fn distance_to(&self, other: Self) -> Self {
        if self < &other {
            other - self
        } else {
            self - other
        }
    }
}

impl Distance for u16 {
    fn distance_to(&self, other: Self) -> Self {
        if self < &other {
            other - self
        } else {
            self - other
        }
    }
}

pub trait MkString {
    fn mk_string_full(&self, start: &str, separator: &str, end: &str) -> String;
    fn mk_string(&self, sep: &str) -> String {
        self.mk_string_full("", sep, "")
    }
}

impl<A: ToString> MkString for Vec<A> {
    fn mk_string_full(&self, start: &str, separator: &str, end: &str) -> String {
        format!(
            "{}{}{}",
            start,
            self.iter().map(|a| a.to_string()).collect::<Vec<String>>().join(separator),
            end,
        )
    }
}

pub trait CloneVectors<A: Clone> {
    fn intercalate_every_n(&mut self, start: A, a: A, end: A, n: usize) -> ();
    fn intercalate_full(&mut self, start: A, a: A, end: A) -> ();
    fn intercalate(&mut self, a: A) -> ();
}

fn intercalate_aux<A: Clone>(v: &mut Vec<A>, a: A, starting_index: usize, n: usize) -> () {
    // Basic idea of algorithm: push the required number of elements (new_items_count). Then,
    // starting from the left most new element, bubble it backwards until it has reached its
    // position. This method is used instead of creating a new vector because we want to avoid
    // cloning the original elements (since we're modifying anyway, no real reason for it).
    // This *does* mean, however that the implementation take O(n^2) instead of O(n), which is bad.
    // TODO create an immutable version of the above, which does this in O(n) but using more clones.
    let len = v.len();
    if len - starting_index == 0 {
        return;
    }
    let new_items_count = (len - starting_index) / n - 1;
    let updated_len = len + new_items_count;
    (0..new_items_count).for_each(|_| v.push(a.clone()));
    let mut reverse_bubble = |i: usize| {
        let distance_from_end = updated_len - i;
        let target_index = i - distance_from_end * n;
        for j in (target_index..i).rev() {
            v.swap(j + 1, j);
        }
    };
    for i in len..updated_len {
        reverse_bubble(i);
    }
}

fn reserved_length_for_intercalated_items<T>(v: &Vec<T>, n: usize) -> usize {
    if v.is_empty() {
        0
    } else {
        v.len() / n - 1
    }
}

// TODO: this should work for all iterables
impl<A: Clone> CloneVectors<A> for Vec<A> {
    fn intercalate_every_n(&mut self, start: A, a: A, end: A, n: usize) -> () {
        self.reserve(2 + reserved_length_for_intercalated_items(self, n));
        self.insert(0, start);
        intercalate_aux(self, a, 1, n);
        self.push(end);
    }
    fn intercalate_full(&mut self, start: A, a: A, end: A) -> () {
        self.reserve(2 + reserved_length_for_intercalated_items(self, 1));
        self.insert(0, start);
        intercalate_aux(self, a, 1, 1);
        self.push(end);
    }
    fn intercalate(&mut self, a: A) -> () {
        self.reserve(reserved_length_for_intercalated_items(self, 1));
        intercalate_aux(self, a, 0, 1);
    }
}

pub trait Vectors<A> {
    // Excepts PartialOrd and caches the function result.
    fn better_sort_by_key<B>(self, f: impl Fn(&A) -> B) -> Vec<A> where B: PartialOrd;
    fn grouped(&self, n: usize) -> Vec<&[A]>;
}

impl<A> Vectors<A> for Vec<A> {
    fn better_sort_by_key<B>(self, f: impl Fn(&A) -> B) -> Vec<A> where B: PartialOrd {
        let mut res = self.into_iter()
            .map(|e| (f(&e), e))
            .collect::<Vec<_>>();
        res.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal));
        res.into_iter().map(|e| e.1).collect()
    }
    fn grouped(&self, n: usize) -> Vec<&[A]> {
        let groups = self.len() / n + if self.len() % n > 0 { 1 } else { 0 };
        let mut result = Vec::with_capacity(groups);
        for i in 0..groups {
            result.push(&self[i * n..min(self.len(), i * n + n)])
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn intercalate_on_empty_does_nothing() {
        let mut v = vec![];
        v.intercalate(1);
        assert!(v.is_empty());
    }

    #[test]
    fn intercalate_on_vector_of_size_1_does_nothing() {
        let mut v = vec![1];
        v.intercalate(1);
        assert_eq!(
            vec![1],
            v,
        );
    }

    #[test]
    fn intercalate_on_vectors_of_size_2() {
        let mut v = vec![1, 2];
        v.intercalate(4);
        assert_eq!(
            vec![1, 4, 2],
            v,
        );
    }

    #[test]
    fn intercalate_on_vectors_of_size_5() {
        let mut v = vec![1, 2, 3, 4, 5];
        v.intercalate(10);
        assert_eq!(
            vec![1, 10, 2, 10, 3, 10, 4, 10, 5],
            v,
        );
    }

    #[test]
    fn intercalate_full_on_empty_wraps() {
        let mut v = vec![];
        v.intercalate_full(10, 42, 20);
        assert_eq!(
            vec![10, 20],
            v,
        );
    }

    #[test]
    fn intercalate_full_on_vector_of_size_1_wraps() {
        let mut v = vec![1];
        v.intercalate_full(10, 42, 20);
        assert_eq!(
            vec![10, 1, 20],
            v,
        );
    }

    #[test]
    fn intercalate_full_on_vectors_of_size_2() {
        let mut v = vec![1, 2];
        v.intercalate_full(10, 42, 20);
        assert_eq!(
            vec![10, 1, 42, 2, 20],
            v,
        );
    }

    #[test]
    fn intercalate_full_on_vectors_of_size_5() {
        let mut v = vec![1, 2, 3, 4, 5];
        v.intercalate_full(10, 42, 20);
        assert_eq!(
            vec![10, 1, 42, 2, 42, 3, 42, 4, 42, 5, 20],
            v,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_empty;
    use super::*;

    #[test]
    fn intercalate() {
        let expected = vec![1, 10, 2, 10, 3];
        let mut actual = vec![1, 2, 3];
        actual.intercalate(10);
        assert_eq!(
            expected,
            actual,
        )
    }

    #[test]
    fn intercalate_full() {
        let expected = vec![10, 1, 20, 2, 20, 3, 30];
        let mut actual = vec![1, 2, 3];
        actual.intercalate_full(10, 20, 30);
        assert_eq!(
            expected,
            actual,
        )
    }

    #[test]
    fn intercalate_every_n() {
        let expected = vec![10, 1, 2, 20, 3, 4, 20, 5, 6, 30];
        let mut actual = vec![1, 2, 3, 4, 5, 6];
        actual.intercalate_every_n(10, 20, 30, 2);
        assert_eq!(
            expected,
            actual,
        )
    }

    #[test]
    fn better_sort_by_key() {
        assert_eq!(
            vec!["my", "name", "moo", "isn't"].better_sort_by_key(|e| e.len()),
            vec!["my", "moo", "name", "isn't"],
        )
    }

    #[test]
    fn grouped_on_empty() {
        let x = Vec::<usize>::new();
        assert_empty!(x.grouped(42));
    }

    #[test]
    fn grouped_on_exact_1() {
        assert_eq!(
            vec![1, 2, 3].grouped(3),
            vec![vec![1, 2, 3]],
        )
    }

    #[test]
    fn grouped_on_exact_2() {
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6].grouped(3),
            vec![vec![1, 2, 3], vec![4, 5, 6]],
        )
    }

    #[test]
    fn grouped_with_extra_element() {
        assert_eq!(
            vec![1, 2, 3, 4, 5, 6, 7].grouped(2),
            vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7]],
        )
    }
}

pub trait Folding<A> {
    // contains would have been a better name, but I'm too tired of the "unstable" compilation errors.
    fn has(&self, a: &A) -> bool where A: Eq;
    fn for_all<P>(&self, p: P) -> bool where P: Fn(&A) -> bool;
    fn exists<P>(&self, p: P) -> bool where P: Fn(&A) -> bool {
        !self.for_all(|e| !p(e))
    }
}

impl<A> Folding<A> for Option<A> {
    fn has(&self, a: &A) -> bool where A: Eq {
        match self {
            None => false,
            Some(s) => s == a,
        }
    }

    fn for_all<P>(&self, p: P) -> bool where P: Fn(&A) -> bool {
        match self {
            None => true,
            Some(s) => p(s),
        }
    }
}
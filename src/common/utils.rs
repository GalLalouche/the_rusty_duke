#[macro_export]
macro_rules! assert_not {
    ($b: expr) => {assert!($b)};
    ($b: expr, $msg: expr) => {assert!($b, $msg)};
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

fn diff_or_zero(x: usize, other: usize) -> usize {
    if x < other {
        x - other
    } else {
        0
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

// TODO: this should work for all iterables
pub trait Vectors<A: Clone> {
    fn intercalate_full(&mut self, start: A, a: A, end: A) -> ();
    fn intercalate(&mut self, a: A) -> ();
}

fn intercalate_aux<A: Clone>(v: &mut Vec<A>, a: A, starting_index: usize) -> () {
    if v.len() - starting_index == 0 {
        return;
    }
    let new_items_count = v.len() - 1 - starting_index;
    (0..new_items_count).for_each(|_| v.push(a.clone()));
    (2 + starting_index..v.len()).step_by(2).rev().for_each(|i| v.swap(i / 2 + starting_index, i));
}

fn reserved_length_for_intercalated_items<T>(v: &Vec<T>) -> usize {
    if v.is_empty() {
        0
    } else {
        v.len() - 1
    }
}

impl<A: Clone> Vectors<A> for Vec<A> {
    fn intercalate_full(&mut self, start: A, a: A, end: A) -> () {
        self.reserve(2 + reserved_length_for_intercalated_items(self));
        self.insert(0, start);
        intercalate_aux(self, a, 1);
        self.push(end);
    }
    fn intercalate(&mut self, a: A) -> () {
        self.reserve(reserved_length_for_intercalated_items(self));
        intercalate_aux(self, a, 0);
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
    fn intercalateon_vectors_of_size_2() {
        let mut v = vec![1, 2];
        v.intercalate(4);
        assert_eq!(
            vec![1, 4, 2],
            v,
        );
    }

    #[test]
    fn intercalateon_vectors_of_size_5() {
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

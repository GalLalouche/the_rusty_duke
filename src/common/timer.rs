use std::cell::RefCell;
use std::collections::HashMap;
use std::time::SystemTime;

thread_local!(pub static GLOBAL_TIMERS: RefCell<HashMap<&'static str, i64>> = RefCell::new(HashMap::new()));

// pub fn time_it<A>(name: &'static str, f: impl FnOnce() -> A) -> A {
//     let start = SystemTime::now();
//     let result = f();
//     let end = SystemTime::now();
//     let total = end.duration_since(start).unwrap().as_nanos() as i64;
//     GLOBAL_TIMERS.with(|map| {
//         let mmap = &mut map.borrow_mut();
//         let new_value = mmap.get(name).unwrap_or(&0) + total;
//         mmap.insert(name, new_value);
//     });
//     result
// }

#[macro_export]
macro_rules! time_it_macro {
    ($name: tt, $expr: tt) => {{
        use std::time::SystemTime;
        let start = SystemTime::now();
        let result = $expr;
        let end = SystemTime::now();
        let total = end.duration_since(start).unwrap().as_nanos() as i64;
        crate::common::timer::GLOBAL_TIMERS.with(|map| {
            let mmap = &mut map.borrow_mut();
            let new_value = mmap.get($name).unwrap_or(&0) + total;
            mmap.insert($name, new_value);
        });
        result
    }}
}

pub fn get_time(name: &'static str) -> Option<i64> {
    GLOBAL_TIMERS.with(|map| map.borrow().get(name).cloned())
}

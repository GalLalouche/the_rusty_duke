use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

thread_local!(pub static GLOBAL_TIMERS: RefCell<HashMap<&'static str, u64>> = RefCell::new(HashMap::new()));

#[macro_export]
macro_rules! time_it_macro {
    ($name: tt, $expr: tt) => {{
        use std::time::SystemTime;
        let start = SystemTime::now();
        let result = $expr;
        let end = SystemTime::now();
        let total = end.duration_since(start).unwrap().as_nanos() as u64;
        crate::common::timer::GLOBAL_TIMERS.with(|map| {
            let mmap = &mut map.borrow_mut();
            let new_value = mmap.get($name).unwrap_or(&0) + total;
            mmap.insert($name, new_value);
        });
        result
    }}
}

pub fn get_time(name: &str) -> Option<Duration> {
    GLOBAL_TIMERS
        .with(|map| map.borrow().get(name).cloned())
        .map(|e| Duration::from_nanos(e as u64))
}

pub fn debug_time(name: &str) -> () {
    let d = get_time(name).unwrap_or(Duration::new(0, 0));
    println!("{} took {:?}", name, d)
}

// Use macro here, make msg optional
pub fn panic_if(b: bool, msg: &String) -> () {
    let x = if b {
        panic!("{}", msg);
    };
}
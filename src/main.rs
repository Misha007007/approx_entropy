//use std::io;

#[macro_use]
extern crate text_io;

fn main() {
    // read until a whitespace and try to convert what was read into an i32
    let i: i64 = read!();
    println!("{}", i);
}

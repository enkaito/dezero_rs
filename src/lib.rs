pub mod functions;
// pub mod layers;
mod macros;
// pub mod optimizers;
pub mod variable;

use std::sync::Mutex;

pub static ENABLE_BACKPROP: Mutex<bool> = Mutex::new(true);

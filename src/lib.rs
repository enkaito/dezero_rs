use ndarray::{Array as A, IxDyn};
pub mod functions;
// pub mod layers;
mod macros;
// pub mod optimizers;
pub mod variable;

pub type Array = A<f32, IxDyn>;

use std::sync::Mutex;

pub static ENABLE_BACKPROP: Mutex<bool> = Mutex::new(true);

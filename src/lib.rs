pub mod array;
pub mod functions;
pub mod macros;
pub mod variable;

use std::sync::Mutex;

pub use array::Array;
pub use variable::VBox;

pub static ENABLE_BACKPROP: Mutex<bool> = Mutex::new(true);

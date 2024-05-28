mod macros;
mod ops;
mod vbox;
mod weak_vbox;

use crate::functions::FnBox;
use ndarray::ArrayD;
pub use vbox::VBox;
pub use weak_vbox::WeakVBox;

pub struct Variable {
    array: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    creator: Option<FnBox>,
    generation: u32,
}

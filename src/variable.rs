mod ops;
mod vbox;
mod weak_vbox;

use crate::array::Array;
use crate::functions::FuncBox;
pub use vbox::VBox;
pub use weak_vbox::WeakVBox;

pub struct Variable {
    array: Array,
    grad: Option<Array>,
    creator: Option<FuncBox>,
    generation: u32,
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        &$crate::variable::VBox::new($crate::array::Array::new(vec![$x as f32], vec![]))
    };
}

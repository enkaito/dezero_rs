#[macro_export]
macro_rules! scaler {
    ($x: expr) => {
        &$crate::variable::VBox::new($crate::array::Array::new(vec![$x as f32], vec![]))
    };
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        &$crate::variable::VBox::new($x)
    };
}

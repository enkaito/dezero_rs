#[macro_export]
macro_rules! scaler {
    ($data: expr) => {
        &$crate::variable::VBox::new(crate::array0!($data))
    };
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        &$crate::variable::VBox::new($x)
    };
}

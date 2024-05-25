#[macro_export]
macro_rules! eval {
    () => {
        *$crate::ENABLE_BACKPROP.lock().unwrap() = false;
    };
}

#[macro_export]
macro_rules! train {
    () => {
        *$crate::ENABLE_BACKPROP.lock().unwrap() = true;
    };
}

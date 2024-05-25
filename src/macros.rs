#[macro_export]
macro_rules! eval {
    ($body: block) => {
        *$crate::ENABLE_BACKPROP.lock().unwrap() = false;
        $body;
        *$crate::ENABLE_BACKPROP.lock().unwrap() = true;
    };
}

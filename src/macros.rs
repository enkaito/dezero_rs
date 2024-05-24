#[macro_export]
macro_rules! square {
    ($x: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Square);
        func.call(&[$x.clone()], true)[0].clone()
    }};

    ($x: expr, $backprop: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Square);
        func.call(&[$x.clone()], $backprop)[0].clone()
    }};
}

#[macro_export]
macro_rules! exp {
    ($x: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Exp);
        func.call(&[$x.clone()], true)[0].clone()
    }};

    ($x: expr, $backprop: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Exp);
        func.call(&[$x.clone()], $backprop)[0].clone()
    }};
}

#[macro_export]
macro_rules! add {
    ($x: expr, $y: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Add);
        func.call(&[$x.clone(), $y.clone()], true)[0].clone()
    }};

    ($x: expr, $y: expr, $backprop: expr) => {{
        let func = $crate::functions::Function::new($crate::functions::FType::Add);
        func.call(&[$x.clone(), $y.clone()], $backprop)[0].clone()
    }};
}

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

#[macro_export]
macro_rules! array {
    ($data: expr, $shape: expr) => {
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn($shape),
            $data.into_iter().map(|x| x as f32).collect(),
        )
        .unwrap()
    };
}

#[macro_export]
macro_rules! array0 {
    ($data: expr) => {
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[]), vec![$data as f32]).unwrap()
    };
}

#[macro_export]
macro_rules! array1 {
    ($data: expr) => {
        ndarray::Array::from_shape_vec(
            ndarray::IxDyn(&[$data.len()]),
            $data.map(|e| e as f32).collect(),
        )
        .unwrap()
    };
}

#[macro_export]
macro_rules! array2 {
    ($data: expr) => {
        ndarray::Array::from_shape_vec(ndarray::IxDyn(&[$data.len(), $data[0].len()]), {
            let mut tmp = Vec::new();
            for row in $data {
                for e in row {
                    tmp.push(e as f32)
                }
            }
            tmp
        })
        .unwrap()
    };
}

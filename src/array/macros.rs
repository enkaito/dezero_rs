#[macro_export]
macro_rules! array0 {
    ($data: expr) => {
        $crate::array::Array::new(vec![$data as f32], Vec::new())
    };
}

#[macro_export]
macro_rules! array1 {
    ($data: expr) => {
        $crate::array::Array::new(
            $data.into_iter().map(|x| x as f32).collect(),
            vec![$data.len()],
        )
    };
}

#[macro_export]
macro_rules! array2 {
    ($data: expr) => {{
        let n = $data.len();
        let m = $data[0].len();
        let flatten: Vec<f32> = $data.into_iter().flatten().map(|x| x as f32).collect();
        let shape = vec![n, m];
        $crate::array::Array::new(flatten, shape)
    }};
}

#[macro_export]
macro_rules! array_with_shape {
    ($data: expr, $shape: expr) => {
        $crate::array::Array::new(
            $data.into_iter().map(|x| x as f32).collect(),
            $shape.into_iter().collect::<Vec<usize>>(),
        )
    };
}

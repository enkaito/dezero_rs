mod array_ops;
use std::{
    fmt::Display,
    iter,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = self.to_string();
        write!(f, "{}", string)
    }
}

impl Array {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Array {
        if data.len() != shape.iter().product() {
            panic!("The data and the shape are inconsistent")
        }
        Array { data, shape }
    }

    pub fn zeros(shape: &Vec<usize>) -> Array {
        let data = iter::repeat(0.).take(shape.iter().product()).collect();
        Array::new(data, shape.clone())
    }

    pub fn ones(shape: &Vec<usize>) -> Array {
        let data = iter::repeat(1.).take(shape.iter().product()).collect();
        Array::new(data, shape.clone())
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn get_shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn set_data(&mut self, new: Array) {
        let data = &mut self.data;
        let new_data = new.data;
        for (old, new) in data.iter_mut().zip(new_data.into_iter()) {
            *old = new
        }
    }
}

#[macro_export]
macro_rules! array0 {
    ($data: expr) => {
        $crate::array::Array::new(vec![$data as f32], Vec::new())
    };
}

macro_rules! array1 {
    ($data: expr) => {
        $crate::array::Array::new(Vec::from($data), Vec::new($data.len()))
    };
}

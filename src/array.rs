mod macros;
mod ops;
mod utils;

use rand::{distributions::Standard, Rng};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: Vec<f32>,
    shape: Vec<usize>,
    size: usize,
}

impl Array {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Array {
        let size = data.len();
        if size != shape.iter().product() {
            panic!("The data and the shape are inconsistent")
        }
        let size = data.len();
        Array { data, shape, size }
    }

    pub fn zeros(shape: &[usize]) -> Array {
        let size = shape.iter().product();
        let data = vec![0.; size];
        Array {
            data,
            shape: shape.to_vec(),
            size,
        }
    }

    pub fn ones(shape: &[usize]) -> Array {
        let size = shape.iter().product();
        let data = vec![1.; size];
        Array {
            data,
            shape: shape.to_vec(),
            size,
        }
    }

    pub fn rand(shape: &[usize]) -> Array {
        let size = shape.iter().product();
        let rng = rand::thread_rng();
        let data = rng.sample_iter(Standard).take(size).collect();
        Array {
            data,
            shape: shape.to_vec(),
            size,
        }
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

    pub fn to_string(&self, depth: usize) -> String {
        array_to_string(&self.data, &self.shape, depth)
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = array_to_string(&self.data, &self.shape, 0);
        write!(f, "{}", string)
    }
}

fn array_to_string(data: &[f32], shape: &[usize], depth: usize) -> String {
    match shape.len() {
        0 => data[0].to_string(),
        1 => format!("{:?}", data),
        _ => {
            let mut acc = "[".to_string();
            acc += &data
                .chunks(data.len() / shape[0])
                .map(|row| array_to_string(row, &shape[1..], depth + 1))
                .collect::<Vec<_>>()
                .join(&format!("\n{}", " ".repeat(depth + 1)));
            acc + "]"
        }
    }
}

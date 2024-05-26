use super::Array;
use core::panic;
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! inner {
    ($x: expr, $y: expr) => {
        $x.zip($y).fold(0., |acc, (x, y)| acc + x * y)
    };
}

impl Array {
    pub fn exp(&self) -> Array {
        let data = self.data.iter().map(|a| a.exp()).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn sin(&self) -> Array {
        let data = self.data.iter().map(|a| a.exp()).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn cos(&self) -> Array {
        let data = self.data.iter().map(|a| a.cos()).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn powi(&self, n: i32) -> Array {
        let data = self.data.iter().map(|a| a.powi(n)).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn powf(&self, n: f32) -> Array {
        let data = self.data.iter().map(|a| a.powf(n)).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn sum(&self) -> Array {
        Array {
            data: vec![self.data.iter().sum()],
            shape: vec![],
            size: 1,
        }
    }

    pub fn sum_to(self, shape: &[usize]) -> Array {
        if self.shape == shape {
            return self;
        }

        let Some(lead) = self.shape.len().checked_sub(shape.len()) else {
            panic!("failed to sum {:?} to {:?}", shape, self.shape)
        };

        let tmp = vec![1; lead];
        let new_shape = tmp
            .into_iter()
            .chain(shape.iter().cloned())
            .collect::<Vec<_>>();

        let mut axes = Vec::new();
        for (axis, (i, j)) in self.shape.iter().zip(new_shape.iter()).enumerate() {
            match (i, j) {
                (i, j) if i == j => {}
                (_, 1) => {
                    axes.push(axis);
                }
                _ => panic!("failed to sum {:?} to {:?}", self.shape, shape),
            }
        }

        let mut data = self.data;
        let dim = self.shape.len();

        let mut steps = vec![1];
        for x in self.shape.iter().rev() {
            steps.push(steps.last().unwrap() * x);
        }

        for axis in axes {
            let mut new_data = Vec::new();
            let step = steps[dim - axis - 1];
            for i in 0..(data.len() / steps[dim - axis]) {
                for j in 0..step {
                    new_data.push(
                        data[steps[dim - axis] * i + j..]
                            .iter()
                            .step_by(step)
                            .take(self.shape[axis])
                            .sum(),
                    )
                }
            }
            data = new_data;
        }

        Array::new(data, shape.to_vec())
    }

    pub fn broadcast_to(self, shape: &[usize]) -> Array {
        if self.shape == shape {
            return self;
        }
        let Some(lead) = shape.len().checked_sub(self.shape.len()) else {
            panic!("failed to broadcast {:?} to {:?}", self.shape, shape)
        };

        let tmp = vec![1; lead];
        let old_shape = tmp
            .into_iter()
            .chain(self.shape.clone())
            .collect::<Vec<_>>();

        let mut axes = Vec::new();
        let mut dups = Vec::new();
        for (axis, (i, j)) in old_shape.iter().zip(shape.iter()).enumerate() {
            match (i, j) {
                (i, j) if i == j => {}
                (1, &dup) => {
                    axes.push(axis);
                    dups.push(dup);
                }
                _ => panic!("failed to broadcast {:?} to {:?}", self.shape, shape),
            }
        }

        let mut data = self.data;
        let dim = shape.len();

        let mut chunk_sizes = vec![1];
        for x in shape {
            chunk_sizes.push(chunk_sizes.last().unwrap() * x);
        }

        for (&axis, &dup) in axes.iter().rev().zip(dups.iter().rev()) {
            data = data
                .chunks(chunk_sizes[dim - axis - 1])
                .map(|c| c.repeat(dup))
                .flatten()
                .collect();
        }

        Array::new(data, shape.to_vec())
    }

    pub fn reshape(self, new_shape: &[usize]) -> Array {
        let new_size = new_shape.iter().product();
        if self.size != new_size {
            panic!("Cannot convert {:?} to {:?}", self.shape, new_shape)
        }
        Array {
            data: self.data,
            shape: new_shape.to_vec(),
            size: self.size,
        }
    }

    pub fn transpose(self) -> Array {
        match self.shape.len() {
            0 | 1 => self,
            2 => self.transpose2d(),
            _ => todo!("transpose for array with dim > 3 is not implemented"),
        }
    }

    fn transpose2d(self) -> Array {
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(self.size);
        for i in 0..n {
            for j in 0..m {
                data.push(self.data[n * j + i])
            }
        }
        Array {
            data,
            shape: vec![n, m],
            size: m * n,
        }
    }

    pub fn dot(&self, rhs: &Array) -> Array {
        match (self.shape.len(), rhs.shape.len()) {
            (0, _) | (_, 0) => self * rhs,
            (_, 1) => {
                let mut shape = self.shape.clone();
                let m = shape.pop().unwrap();
                assert_eq!(
                    m, rhs.shape[0],
                    "Invalid shapes of the array to multiply.\nlhs: {:?}\nrhs: {:?}",
                    self.shape, rhs.shape
                );
                let data = self
                    .data
                    .chunks(m)
                    .map(|x| inner!(x.iter(), rhs.data.iter()))
                    .collect::<Vec<f32>>();
                Array::new(data, shape)
            }
            (2, 2) => {
                let (l, m, m_, n) = (self.shape[0], self.shape[1], rhs.shape[0], rhs.shape[1]);
                assert_eq!(
                    m, m_,
                    "{}x{} and {}x{} matrices cannot be multiplied",
                    self.shape[0], self.shape[1], rhs.shape[0], rhs.shape[1]
                );

                let lhs = &self.data;
                let rhs = &rhs.data;
                let mut data = Vec::with_capacity(l * n);
                for i in 0..l {
                    for j in 0..n {
                        data.push(inner!(
                            lhs[m * i..].iter().take(m),
                            rhs[j..].iter().step_by(n)
                        ));
                    }
                }
                Array::new(data, vec![l, n])
            }
            (_, d) if d >= 2 => {
                let (k, l, m): (_, _, usize) = (
                    self.shape[0],
                    self.shape[1],
                    self.shape[2..].iter().product(),
                );
                let (n, m_, o): (_, usize, _) = (
                    rhs.shape[0],
                    rhs.shape[1..d - 1].iter().product(),
                    rhs.shape[d - 1],
                );
                assert_eq!(
                    m, m_,
                    "Invalid shapes of the array to multiply.\nlhs: {:?}\nrhs: {:?}",
                    self.shape, rhs.shape
                );
                let lhs = &self.data;
                let rhs = &rhs.data;

                let mut data = Vec::with_capacity(k * l * n * o);
                for pq in 0..k * l {
                    for r in 0..n {
                        for s in 0..o {
                            data.push(inner!(
                                lhs[pq * m..].iter().take(m),
                                rhs[r * n * m + s..].iter().step_by(o)
                            ))
                        }
                    }
                }

                let shape = vec![k, l, n, o];
                Array::new(data, shape)
            }
            _ => todo!(
                "Invalid shapes of the array to multiply.\nlhs: {:?}\nrhs: {:?}",
                self.shape,
                rhs.shape
            ),
        }
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                if self.shape.is_empty() {
                    return f32::$fname(self.data[0], rhs);
                }
                if rhs.shape.is_empty() {
                    return Array::$fname(self, rhs.data[0]);
                }
                if self.shape != rhs.shape {
                    panic!(
                        "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                        self, rhs,
                    );
                }
                let data = self
                    .data
                    .iter()
                    .zip(rhs.data.iter())
                    .map(|(x, y)| f32::$fname(*x, y))
                    .collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait<Array> for &Array {
            type Output = Array;
            fn $fname(self, rhs: Array) -> Self::Output {
                Array::$fname(self.clone(), rhs)
            }
        }

        impl $trait<&Array> for Array {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                Array::$fname(self, rhs.clone())
            }
        }

        impl $trait for &Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                Array::$fname(self.clone(), rhs.clone())
            }
        }

        impl $trait<f32> for Array {
            type Output = Array;
            fn $fname(self, rhs: f32) -> Self::Output {
                let data = self.data.into_iter().map(|x| f32::$fname(x, rhs)).collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait<Array> for f32 {
            type Output = Array;
            fn $fname(self, rhs: Array) -> Self::Output {
                let data = rhs.data.into_iter().map(|x| f32::$fname(self, x)).collect();
                Array {
                    data,
                    shape: rhs.shape.clone(),
                    size: rhs.size,
                }
            }
        }

        impl $trait<f32> for &Array {
            type Output = Array;
            fn $fname(self, rhs: f32) -> Self::Output {
                let data = self.data.iter().map(|x| f32::$fname(*x, rhs)).collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait<&Array> for f32 {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                let data = rhs.data.iter().map(|x| f32::$fname(self, x)).collect();
                Array {
                    data,
                    shape: rhs.shape.clone(),
                    size: rhs.size,
                }
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl Neg for Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        let data = self.data.iter().map(|a| -a).collect();
        Array::new(data, self.shape.clone())
    }
}

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

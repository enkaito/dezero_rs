use super::Array;
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! inner {
    ($x: expr, $y: expr) => {
        $x.zip($y).fold(0., |acc, (x, y)| acc + x * y)
    };
}

fn sum(data: &[f32], shape: &[usize], axis: usize) -> Vec<f32> {
    match axis {
        0 => {
            let len = data.len() / shape[0];
            let mut res = Vec::with_capacity(len);
            for i in 0..len {
                let sum = (0..shape[0]).fold(0., |acc, j| acc + data[i + len * j]);
                res.push(sum);
            }
            res
        }
        _ => data
            .chunks(data.len() / shape[0])
            .map(|x| sum(x, &shape[1..], axis - 1))
            .flatten()
            .collect(),
    }
}

fn broadcast(data: &[f32], shape: &[usize], axis: usize, dup: usize) -> Vec<f32> {
    match axis {
        0 => data
            .iter()
            .cycle()
            .take(data.len() * dup)
            .cloned()
            .collect(),
        _ => data
            .chunks(data.len() / shape[0])
            .map(|x| broadcast(x, &shape[1..], axis - 1, dup))
            .flatten()
            .collect(),
    }
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

    pub fn sum_axis(&self, axis: usize, keep_dims: bool) -> Array {
        if self.shape.len() < axis {
            panic!("Axis {} doesn't exist in:\n{}", axis, self)
        }
        let new_data = sum(&self.data, &self.shape, axis);

        let mut new_shape = self.shape.clone();
        if keep_dims {
            new_shape[axis] = 1;
        } else {
            new_shape.remove(axis);
        }

        Array::new(new_data, new_shape)
    }

    pub fn broadcast(&self, axis: usize, keep_dims: bool, dup: usize) -> Array {
        if self.shape.len() < axis {
            panic!("Failed to expand axis {} in:\n{}", axis, self);
        }

        let new_data = broadcast(&self.data, &self.shape, axis, dup);

        let mut new_shape = self.shape.clone();
        if keep_dims {
            if new_shape[axis] != 1 {
                panic!("Broadcast failed trying to expand the dimension with size != 1")
            }
            new_shape[axis] = dup;
        } else {
            new_shape.insert(axis, dup);
        }

        Array::new(new_data, new_shape)
    }

    pub fn broadcast_scaler(&self, shape: &[usize]) -> Array {
        let size = shape.iter().product();
        let data = vec![self.data[0]; size];
        Array {
            data,
            shape: shape.to_vec(),
            size,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Array {
        let new_size = new_shape.iter().product();
        if self.size != new_size {
            panic!("Cannot convert {:?} to {:?}", self.shape, new_shape)
        }
        Array {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            size: self.size,
        }
    }

    pub fn transpose(&self) -> Array {
        match self.shape.len() {
            0 | 1 => self.clone(),
            2 => self.transpose2d(),
            _ => todo!("transpose for array with dim > 3 is not implemented"),
        }
    }

    fn transpose2d(&self) -> Array {
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
                            rhs[j..].iter().step_by(m)
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

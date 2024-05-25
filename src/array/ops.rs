use super::Array;
use std::ops::{Add, Div, Mul, Neg, Sub};

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

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Array {
        let new_size = new_shape.iter().product();
        if self.size != new_size {
            panic!("Cannot convert {:?} to {:?}", self.shape, new_shape)
        }
        Array {
            data: self.data.clone(),
            shape: new_shape,
            size: self.size,
        }
    }

    // pub fn transpose(&self) -> Array {}

    pub fn all_close(&self, rhs: &Array, tol: f32) -> bool {
        (self - rhs)
            .data
            .iter()
            .try_for_each(|x| if x < &tol { Some(()) } else { None })
            .is_some()
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                if self.shape != rhs.shape {
                    panic!(
                        "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                        self, rhs,
                    );
                }
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.data.into_iter())
                    .map(|(x, y)| f32::$fname(x, y))
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
                if self.shape != rhs.shape {
                    panic!(
                        "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                        self, rhs,
                    );
                }
                let data = self
                    .data
                    .iter()
                    .zip(rhs.data.into_iter())
                    .map(|(x, y)| f32::$fname(y, x))
                    .collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait<&Array> for Array {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                if self.shape != rhs.shape {
                    panic!(
                        "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                        self, rhs,
                    );
                }
                let data = self
                    .data
                    .into_iter()
                    .zip(rhs.data.iter())
                    .map(|(x, y)| f32::$fname(x, y))
                    .collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait for &Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
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
                let data = self.data.iter().map(|x| f32::$fname(rhs, x)).collect();
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

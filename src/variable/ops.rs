use super::VBox;
use crate::{
    functions::{self as F},
    scaler,
};
use std::ops::{Add, Div, Mul, Neg, Sub};

impl VBox {
    pub fn powi(&self, c: i32) -> VBox {
        let func = F::Pow::new(c as f32);
        F::call(func, &[self.clone()])
    }

    pub fn pow(&self, c: f32) -> VBox {
        let func = F::Pow::new(c);
        F::call(func, &[self.clone()])
    }

    pub fn exp(&self) -> VBox {
        let func = F::Exp::new();
        F::call(func, &[self.clone()])
    }

    pub fn reshape(&self, shape: Vec<usize>) -> VBox {
        let func = F::Reshape::new(self.get_shape(), shape);
        F::call(func, &[self.clone()])
    }

    pub fn transpose(&self) -> VBox {
        let func = F::Transpose::new();
        F::call(func, &[self.clone()])
    }

    pub fn sum(&self) -> VBox {
        let func = F::Sum::new(self.get_shape());
        F::call(func, &[self.clone()])
    }

    pub fn sum_to(&self, shape: &[usize]) -> VBox {
        let func = F::SumTo::new(self.get_shape(), shape.to_vec());
        F::call(func, &[self.clone()])
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> VBox {
        let func = F::BroadcastTo::new(self.get_shape(), shape.to_vec());
        F::call(func, &[self.clone()])
    }

    pub fn dot(&self, rhs: &VBox) -> VBox {
        let func = F::Dot::new();
        F::call(func, &[self.clone(), rhs.clone()])
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                let func = F::$trait::new(self.get_shape(), rhs.get_shape());
                F::call(func, &[self, rhs]).clone()
            }
        }

        impl $trait<VBox> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                VBox::$fname(self.clone(), rhs)
            }
        }

        impl $trait<&VBox> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                VBox::$fname(self, rhs.clone())
            }
        }

        impl $trait for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: Self) -> Self::Output {
                VBox::$fname(self.clone(), rhs.clone())
            }
        }

        impl $trait<f32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(scaler!(rhs))
            }
        }

        impl $trait<f32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<i32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<i32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Mul, mul);
impl_op!(Sub, sub);
impl_op!(Div, div);

impl Neg for VBox {
    type Output = VBox;
    fn neg(self) -> Self::Output {
        let func = F::Neg::new();
        F::call(func, &[self])
    }
}

impl Neg for &VBox {
    type Output = VBox;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

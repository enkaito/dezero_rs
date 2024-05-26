use super::VBox;
use crate::{
    functions::{FType, Function},
    scaler,
};
use std::ops::{Add, Div, Mul, Neg, Sub};

impl VBox {
    pub fn powi(&self, c: i32) -> VBox {
        let func = Function::new(FType::Powi(c));
        func.call(&[self.clone()])[0].clone()
    }

    pub fn powf(&self, c: f32) -> VBox {
        let func = Function::new(FType::Powf(c));
        func.call(&[self.clone()])[0].clone()
    }

    pub fn exp(&self) -> VBox {
        let func = Function::new(FType::Exp);
        func.call(&[self.clone()])[0].clone()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> VBox {
        let func = Function::new(FType::Reshape(self.get_shape(), shape));
        func.call(&[self.clone()])[0].clone()
    }

    pub fn transpose(&self) -> VBox {
        let func = Function::new(FType::Transpose);
        func.call(&[self.clone()])[0].clone()
    }

    pub fn sum(&self) -> VBox {
        let func = Function::new(FType::Sum(self.get_shape()));
        func.call(&[self.clone()])[0].clone()
    }

    // pub fn sum_axis(&self, axis: usize, keep_dims: bool) -> VBox {
    //     let func = Function::new(FType::SumAxis(axis, keep_dims, self.get_shape()[axis]));
    //     func.call(&[self.clone()])[0].clone()
    // }

    pub fn dot(&self, rhs: &VBox) -> VBox {
        let func = Function::new(FType::Matmul);
        func.call(&[self.clone(), rhs.clone()])[0].clone()
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                let func = Function::new(FType::$trait);
                func.call(&[self, rhs])[0].clone()
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
        let func = Function::new(FType::Neg);
        func.call(&[self])[0].clone()
    }
}

impl Neg for &VBox {
    type Output = VBox;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

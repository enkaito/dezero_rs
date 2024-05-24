use super::{FType, Function, VBox};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                let func = Function::new(FType::$trait);
                func.call(&[self, rhs], true)[0].clone()
            }
        }

        impl $trait<VBox> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                self.clone().$fname(rhs)
            }
        }

        impl $trait<&VBox> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                self.$fname(rhs.clone())
            }
        }

        impl $trait for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: Self) -> Self::Output {
                self.clone().$fname(rhs.clone())
            }
        }

        impl $trait<f32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(crate::var!(rhs))
            }
        }

        impl $trait<f32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(crate::var!(rhs))
            }
        }

        impl $trait<VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::var!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::var!(self).$fname(rhs)
            }
        }

        impl $trait<i32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::var!(rhs))
            }
        }

        impl $trait<i32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::var!(rhs))
            }
        }

        impl $trait<VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::var!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::var!(self).$fname(rhs)
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
        func.call(&[self], true)[0].clone()
    }
}

impl VBox {
    pub fn pow(&self, c: f32) -> VBox {
        let func = Function::new(FType::Pow(c.into()));
        func.call(&[self.clone()], true)[0].clone()
    }
}

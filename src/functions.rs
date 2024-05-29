use ndarray::ArrayD;

use crate::variable::{VBox, WeakVBox};
use std::{hash::Hash, rc::Rc};

pub fn call(mut f: impl Function + 'static, input: Vec<VBox>) -> VBox {
    let y = f.forward(input.clone());
    let output = VBox::new(y);
    f.set_inputs(input.clone());
    f.set_output(output.clone().downgrade());

    if *crate::ENABLE_BACKPROP.lock().unwrap() {
        f.set_generation(input.iter().map(|x| x.get_gen()).max().unwrap());
        let to_f = Rc::new(f);
        output.set_creator(FnBox(to_f.clone()));
    }
    output
}

fn sum_to(lhs: &ArrayD<f32>, shape: &[usize]) -> ArrayD<f32> {
    let ndim = shape.len();
    let lead = lhs.ndim() - ndim;
    todo!()
}

// pub fn linear(x: &VBox, w: &VBox, b: Option<&VBox>) -> VBox {
//     let bias = b.is_some();
//     let func = Linear::new(bias);
//     if bias {
//         call(func, &[x.clone(), w.clone(), b.unwrap().clone()])
//     } else {
//         call(func, &[x.clone(), w.clone()])
//     }
// }

// pub fn sigmoid(x: &VBox) -> VBox {
//     let func = Sigmoid::new();
//     call(func, &[x.clone()])
// }

// pub fn relu(x: &VBox) -> VBox {
//     let func = ReLU::new();
//     call(func, &[x.clone()])
// }

// pub fn mean_squared_error(x: &VBox, y: &VBox) -> VBox {
//     let func = MeanSquaredError::new();
//     call(func, &[x.clone(), y.clone()])
// }

// pub fn softmax(x: &VBox, axis: usize) -> VBox {
//     let func = Softmax::new(axis);
//     call(func, &[x.clone()])
// }

// pub fn cross_entropy_loss(x: &VBox, t: &VBox) -> VBox {
//     let func = CrossEnrtopy::new();
//     call(func, &[x.clone(), t.clone()])
// }

pub trait Function {
    fn get_generation(&self) -> u32;
    fn get_inputs(&self) -> Vec<VBox>;
    fn get_output(&self) -> WeakVBox;

    fn set_generation(&mut self, gen: u32);
    fn set_inputs(&mut self, inputs: Vec<VBox>);
    fn set_output(&mut self, output: WeakVBox);

    fn forward(&self, x: Vec<VBox>) -> ArrayD<f32>;
    fn backward(&self, gy: ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

macro_rules! impl_getters_setters {
    ($n: literal) => {
        fn get_generation(&self) -> u32 {
            self.generation
        }

        fn get_inputs(&self) -> Vec<VBox> {
            self.inputs.clone().unwrap()
        }

        fn get_output(&self) -> WeakVBox {
            self.output.clone().unwrap()
        }

        fn set_generation(&mut self, gen: u32) {
            self.generation = gen
        }

        fn set_inputs(&mut self, inputs: Vec<VBox>) {
            self.inputs = Some(inputs)
        }

        fn set_output(&mut self, output: WeakVBox) {
            self.output = Some(output)
        }
    };
}

macro_rules! define {
    ($name: ident, $arity: literal, $($key: ident: $type: ty),*) => {
        pub struct $name {
            inputs: Option<Vec<VBox>>,
            output: Option<WeakVBox>,
            generation: u32,
            $(
                $key: $type
            ),*
        }

        impl $name {
            pub fn new($($key: $type),*) -> Self {
                Self {
                    inputs: None,
                    output: None,
                    generation: 0,
                    $(
                        $key
                    ),*
                }
            }
        }
    };
}

macro_rules! define_binop {
    ($name: ident) => {
        define!($name, 2, shape0: Vec<usize>, shape1: Vec<usize>);
    };
}

define_binop!(Add);
impl Function for Add {
    impl_getters_setters!(2);
    fn forward(&self, x: Vec<VBox>) -> ArrayD<f32> {
        &x[0].get_array() + &x[1].get_array()
    }
    fn backward(&self, gy: ArrayD<f32>) -> Vec<ArrayD<f32>> {
        // let gx0 = gy;
        // let gx1 = gy;
        // if self.shape0 != self.shape1 {
        //     vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        // } else {
        //     [gy, gy]
        // }
        vec![gy.clone(), gy]
    }
}

// define_binop!(Sub);
// impl Function for Sub {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         &x[0] - &x[1]
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let gx0 = gy.clone();
//         let gx1 = -gy;
//         if self.shape0 != self.shape1 {
//             vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
//         } else {
//             vec![gx0, gx1]
//         }
//     }
// }

// define_binop!(Mul);
// impl Function for Mul {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         &x[0] * &x[1]
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         let gx0 = &x[1] * &gy;
//         let gx1 = &x[0] * gy;
//         if self.shape0 != self.shape1 {
//             vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
//         } else {
//             vec![gx0, gx1]
//         }
//     }
// }

// define_binop!(Div);
// impl Function for Div {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         &x[0] / &x[1]
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         let gx0 = &gy / &x[1];
//         let gx1 = -&x[0] / (&x[1] * &x[1]) * gy;
//         if self.shape0 != self.shape1 {
//             vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
//         } else {
//             vec![gx0, gx1]
//         }
//     }
// }

// define!(Neg,);
// impl Function for Neg {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         -x[0].clone()
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![-gy]
//     }
// }

// define!(Powi, n: i32);
// impl Function for Powi {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].powi(self.n)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
//         vec![self.n as f32 * x.powi(self.n - 1) * gy]
//     }
// }

// define!(Powf, c: f32);
// impl Function for Powf {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].powf(self.c)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
//         vec![self.c * x.powf(self.c - 1.) * gy]
//     }
// }

// define!(Exp,);
// impl Function for Exp {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].exp()
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
//         vec![x.exp() * gy]
//     }
// }

// define!(Reshape, shape_in: Vec<usize>, shape_out: Vec<usize>);
// impl Function for Reshape {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].clone().reshape(&self.shape_out)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![gy.reshape(&self.shape_in)]
//     }
// }

// define!(Transpose,);
// impl Function for Transpose {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].clone().transpose()
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![gy.transpose()]
//     }
// }

// define!(Sum, shape: Vec<usize>);
// impl Function for Sum {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].sum()
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![gy.broadcast_to(&self.shape)]
//     }
// }

// define!(SumTo, shape_in: Vec<usize>, shape_out: Vec<usize>);
// impl Function for SumTo {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].clone().sum_to(&self.shape_out)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![gy.broadcast_to(&self.shape_in)]
//     }
// }

// define!(BroadcastTo, shape_in: Vec<usize>, shape_out: Vec<usize>);
// impl Function for BroadcastTo {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].clone().broadcast_to(&self.shape_out)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         vec![gy.sum_to(&self.shape_in)]
//     }
// }

// define!(Matmul,);
// impl Function for Matmul {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].matmul(&x[1])
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         vec![
//             gy.matmul(&x[1].clone().transpose()),
//             x[0].clone().transpose().matmul(&gy),
//         ]
//     }
// }

// define!(Linear, bias: bool);
// impl Function for Linear {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         let t = x[0].matmul(&x[1]);
//         if self.bias {
//             t + &x[2]
//         } else {
//             t
//         }
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         let gx = gy.matmul(&x[1].clone().transpose());
//         let gw = x[0].clone().transpose().matmul(&gy);
//         if self.bias {
//             vec![gx, gw, gy.sum_to(&x[2].get_shape())]
//         } else {
//             vec![gx, gw]
//         }
//     }
// }

// define!(Sigmoid,);
// impl Function for Sigmoid {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         (&x[0] * 0.5).tanh() * 0.5 + 0.5
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let y = self.output.as_ref().unwrap().get_array();
//         vec![gy * &y * (1. - y)]
//     }
// }

// define!(ReLU,);
// impl Function for ReLU {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         x[0].relu_max(0.)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
//         vec![x.relu_mask(&gy, 0.)]
//     }
// }

// define!(MeanSquaredError,);
// impl Function for MeanSquaredError {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         let diff = &x[0] - &x[1];
//         diff.powi(2).sum() / diff.size() as f32
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         let diff = &x[0] - &x[1];
//         let gx = gy * &diff * (2. / diff.size() as f32);
//         vec![gx.clone(), -gx]
//     }
// }

// define!(Softmax, axis: usize);
// impl Function for Softmax {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         let y = (&x[0] - x[0].max(self.axis)).exp();
//         &y / y.sum_with_axis(self.axis)
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let y = self.output.as_ref().unwrap().get_array();
//         let gx = &y * gy;
//         let sumdx = &gx.sum_with_axis(self.axis);
//         vec![gx - y * sumdx]
//     }
// }

// define!(CrossEnrtopy,);
// impl Function for CrossEnrtopy {
//     impl_getters_setters!();
//     fn forward(&self, x: Vec<Array>) -> Array {
//         -x[0].clip(1e-15, 1.).ln().matmul(&x[1].transpose()).sum()
//     }
//     fn backward(&self, gy: Array) -> Vec<Array> {
//         let x: Vec<Array> = self
//             .inputs
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|x| x.get_array())
//             .collect();
//         let cliped_x = &x[0].clip(1e-15, 1.);
//         let gx = -&x[1] / cliped_x;
//         let gt = -cliped_x.ln();
//         vec![gx * &gy, gt * &gy]
//     }
// }

#[derive(Clone)]
pub struct FnBox(Rc<dyn Function>);

impl FnBox {
    pub fn get_gen(&self) -> u32 {
        self.0.get_generation()
    }

    pub fn get_inputs(&self) -> Vec<VBox> {
        self.0.get_inputs()
    }

    pub fn get_output(&self) -> WeakVBox {
        self.0.get_output()
    }

    pub fn backward(&self, gy: ArrayD<f32>) -> Vec<ArrayD<f32>> {
        self.0.backward(gy)
    }
}

impl PartialEq for FnBox {
    fn eq(&self, other: &FnBox) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for FnBox {}

impl Hash for FnBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize((Rc::as_ptr(&self.0) as *mut usize) as usize);
    }
}

impl Ord for FnBox {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.get_generation().cmp(&other.0.get_generation())
    }
}

impl PartialOrd for FnBox {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

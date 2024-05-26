use crate::{
    array::Array,
    variable::{VBox, WeakVBox},
};
use std::{hash::Hash, rc::Rc};

pub fn call(mut f: impl Function + 'static, input: &[VBox]) -> Vec<VBox> {
    let x = input.iter().map(|i| i.get_array()).collect();
    let y = f.forward(x);
    let outputs: Vec<VBox> = y.into_iter().map(|y| VBox::new(y)).collect();
    f.set_inputs(input.into());
    f.set_outputs(outputs.iter().map(|o| o.clone().downgrade()).collect());

    if *crate::ENABLE_BACKPROP.lock().unwrap() {
        f.set_generation(input.iter().map(|x| x.get_gen()).max().unwrap());
        let to_f = Rc::new(f);
        for output in outputs.iter() {
            output.set_creator(FuncBox(to_f.clone()))
        }
    }
    outputs
}

pub fn linear(x: &VBox, w: &VBox, b: Option<&VBox>) -> VBox {
    let bias = b.is_some();
    let func = Linear::new(bias);
    if bias {
        call(func, &[x.clone(), w.clone(), b.unwrap().clone()])[0].clone()
    } else {
        call(func, &[x.clone(), w.clone()])[0].clone()
    }
}

pub fn sigmoid(x: &VBox) -> VBox {
    let func = Sigmoid::new();
    call(func, &[x.clone()])[0].clone()
}

pub fn mean_squared_error(x: &VBox, y: &VBox) -> VBox {
    let func = MeanSquaredError::new();
    call(func, &[x.clone(), y.clone()])[0].clone()
}

pub trait Function {
    fn get_generation(&self) -> u32;
    fn get_inputs(&self) -> Vec<VBox>;
    fn get_outputs(&self) -> Vec<WeakVBox>;

    fn set_generation(&mut self, gen: u32);
    fn set_inputs(&mut self, inputs: Vec<VBox>);
    fn set_outputs(&mut self, outputs: Vec<WeakVBox>);

    fn forward(&self, x: Vec<Array>) -> Vec<Array>;
    fn backward(&self, gy: Vec<Array>) -> Vec<Array>;
}

macro_rules! impl_getters_setters {
    () => {
        fn get_generation(&self) -> u32 {
            self.generation
        }

        fn get_inputs(&self) -> Vec<VBox> {
            self.inputs.clone().unwrap()
        }

        fn get_outputs(&self) -> Vec<WeakVBox> {
            self.outputs.clone().unwrap()
        }

        fn set_generation(&mut self, gen: u32) {
            self.generation = gen
        }

        fn set_inputs(&mut self, inputs: Vec<VBox>) {
            self.inputs = Some(inputs)
        }

        fn set_outputs(&mut self, outputs: Vec<WeakVBox>) {
            self.outputs = Some(outputs)
        }
    };
}

macro_rules! define {
    ($name: ident, $($key: ident: $type: ty),*) => {
        pub struct $name {
            inputs: Option<Vec<VBox>>,
            outputs: Option<Vec<WeakVBox>>,
            generation: u32,
            $(
                $key: $type
            ),*
        }

        impl $name {
            pub fn new($($key: $type),*) -> Self {
                Self {
                    inputs: None,
                    outputs: None,
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
        define!($name, shape0: Vec<usize>, shape1: Vec<usize>);
    };
}

define_binop!(Add);
impl Function for Add {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![&x[0] + &x[1]]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let gx0 = gy[0].clone();
        let gx1 = gy[0].clone();
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

define_binop!(Sub);
impl Function for Sub {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![&x[0] - &x[1]]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let gx0 = gy[0].clone();
        let gx1 = -&gy[0];
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

define_binop!(Mul);
impl Function for Mul {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![&x[0] * &x[1]]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx0 = &x[1] * &gy[0];
        let gx1 = &x[0] * &gy[0];
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

define_binop!(Div);
impl Function for Div {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![&x[0] / &x[1]]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx0 = &gy[0] / &x[1];
        let gx1 = -&x[0] / (&x[1] * &x[1]) * &gy[0];
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

define!(Neg,);
impl Function for Neg {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![-x[0].clone()]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![-gy[0].clone()]
    }
}

define!(Pow, c: f32);
impl Function for Pow {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].powf(self.c)]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![self.c * x.powf(self.c - 1.) * &gy[0]]
    }
}

define!(Exp,);
impl Function for Exp {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].exp()]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![x.exp() * &gy[0]]
    }
}

define!(Reshape, shape_in: Vec<usize>, shape_out: Vec<usize>);
impl Function for Reshape {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].clone().reshape(&self.shape_out)]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![gy[0].clone().reshape(&self.shape_in)]
    }
}

define!(Transpose,);
impl Function for Transpose {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].clone().transpose()]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![gy[0].clone().transpose()]
    }
}

define!(Sum, shape: Vec<usize>);
impl Function for Sum {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].sum()]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![gy[0].clone().broadcast_to(&self.shape)]
    }
}

define!(SumTo, shape_in: Vec<usize>, shape_out: Vec<usize>);
impl Function for SumTo {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].clone().sum_to(&self.shape_out)]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![gy[0].clone().broadcast_to(&self.shape_in)]
    }
}

define!(BroadcastTo, shape_in: Vec<usize>, shape_out: Vec<usize>);
impl Function for BroadcastTo {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].clone().broadcast_to(&self.shape_out)]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        vec![gy[0].clone().sum_to(&self.shape_in)]
    }
}

define!(Dot,);
impl Function for Dot {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![x[0].dot(&x[1])]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        vec![
            gy[0].dot(&x[1].clone().transpose()),
            x[0].clone().transpose().dot(&gy[0]),
        ]
    }
}

define!(Linear, bias: bool);
impl Function for Linear {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        let t = x[0].dot(&x[1]);
        if self.bias {
            vec![t + &x[2]]
        } else {
            vec![t]
        }
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx = gy[0].dot(&x[1].clone().transpose());
        let gw = x[0].clone().transpose().dot(&gy[0]);
        if self.bias {
            vec![gx, gw, gy[0].clone().sum_to(&x[2].get_shape())]
        } else {
            vec![gx, gw]
        }
    }
}

define!(Sigmoid,);
impl Function for Sigmoid {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        vec![(&x[0] * 0.5).tanh() * 0.5 + 0.5]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let y = self.outputs.as_ref().unwrap().first().unwrap().get_array();
        vec![&gy[0] * &y * (1. - y)]
    }
}

define!(MeanSquaredError,);
impl Function for MeanSquaredError {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Vec<Array> {
        let diff = &x[0] - &x[1];
        vec![diff.powi(2).sum() / diff.size() as f32]
    }
    fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let diff = &x[0] - &x[1];
        let gx = &gy[0] * &diff * (2. / diff.size() as f32);
        vec![gx.clone(), -gx]
    }
}

#[derive(Clone)]
pub struct FuncBox(Rc<dyn Function>);

impl FuncBox {
    pub fn get_gen(&self) -> u32 {
        self.0.get_generation()
    }

    pub fn get_inputs(&self) -> Vec<VBox> {
        self.0.get_inputs()
    }

    pub fn get_outputs(&self) -> Vec<WeakVBox> {
        self.0.get_outputs()
    }

    pub fn backward(&self, gy: Vec<Array>) -> Vec<Array> {
        self.0.backward(gy)
    }
}

impl PartialEq for FuncBox {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for FuncBox {}

impl Hash for FuncBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize((Rc::as_ptr(&self.0) as *mut usize) as usize);
    }
}

impl Ord for FuncBox {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.get_generation().cmp(&other.0.get_generation())
    }
}

impl PartialOrd for FuncBox {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

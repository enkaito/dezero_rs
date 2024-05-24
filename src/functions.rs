use crate::variable::{VBox, WeakVBox};
use std::{hash::Hash, rc::Rc};

pub enum FType {
    Square,
    Exp,
    Add,
    Mul,
    Neg,
    Sub,
    Div,
    Pow(f32),
}

pub struct Function {
    inputs: Option<Vec<VBox>>,
    outputs: Option<Vec<WeakVBox>>,
    ftype: FType,
    generation: u32,
}

impl Function {
    pub fn new(ftype: FType) -> Function {
        Function {
            inputs: None,
            outputs: None,
            ftype: ftype,
            generation: 0,
        }
    }

    pub fn get_gen(&self) -> u32 {
        self.generation
    }

    pub fn clone_input(&self) -> Vec<VBox> {
        self.inputs.clone().unwrap()
    }

    pub fn clone_output(&self) -> Vec<WeakVBox> {
        self.outputs.clone().unwrap()
    }

    pub fn call(mut self, input: &[VBox], enable_backprop: bool) -> Vec<VBox> {
        let x = input.iter().map(|i| i.get_data()).collect();
        let y = self.forward(x);
        let outputs: Vec<VBox> = y.iter().map(|&y| VBox::new(y)).collect();
        self.inputs = Some(input.into());
        self.outputs = Some(outputs.iter().map(|o| o.clone().downgrade()).collect());

        if enable_backprop {
            self.generation = input.iter().map(|x| x.get_gen()).max().unwrap();
            let to_self = Rc::new(self);
            for output in outputs.iter() {
                output.set_creator(to_self.clone())
            }
        }
        outputs
    }

    fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        match &self.ftype {
            FType::Square => vec![x[0].powi(2)],
            FType::Exp => vec![x[0].exp()],
            FType::Add => vec![x[0] + x[1]],
            FType::Mul => vec![x[0] * x[1]],
            FType::Neg => vec![-x[0]],
            FType::Sub => vec![x[0] - x[1]],
            FType::Div => vec![x[0] / x[1]],
            FType::Pow(c) => vec![x[0].powf(*c)],
        }
    }

    pub fn backward(&self, gy: Vec<f32>) -> Vec<f32> {
        let x: Vec<f32> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_data())
            .collect();
        match &self.ftype {
            FType::Square => vec![2. * x[0] * gy[0]],
            FType::Exp => vec![x[0].exp() * gy[0]],
            FType::Add => vec![gy[0], gy[0]],
            FType::Mul => vec![x[1] * gy[0], x[0] * gy[0]],
            FType::Neg => vec![-gy[0]],
            FType::Sub => vec![gy[0], -gy[0]],
            FType::Div => vec![gy[0] / x[1], -x[0] / x[1].powi(2) * gy[0]],
            FType::Pow(c) => vec![c * x[0].powf(c - 1.) * gy[0]],
        }
    }
}

#[derive(Clone)]
pub struct FuncBox(pub Rc<Function>);

impl PartialEq for FuncBox {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for FuncBox {}

impl Hash for FuncBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Rc::as_ptr(&self.0) as usize);
    }
}

impl Ord for FuncBox {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.get_gen().cmp(&other.0.get_gen())
    }
}

impl PartialOrd for FuncBox {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

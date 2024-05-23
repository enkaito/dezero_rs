use crate::functions::{FuncBox, Function};

use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    rc::{Rc, Weak},
};

struct VariableData {
    data: f32,
    grad: Option<f32>,
    creator: Option<FuncBox>,
    generation: u32,
}

#[derive(Clone)]
pub struct Variable(Rc<RefCell<VariableData>>);

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Variable({}, grad: {})",
            self.get_data(),
            match self.get_option_grad() {
                Some(g) => g.to_string(),
                None => "None".to_string(),
            }
        )
    }
}

impl Variable {
    pub fn new(data: f32) -> Variable {
        Variable(Rc::new(RefCell::new(VariableData {
            data,
            grad: None,
            creator: None,
            generation: 0,
        })))
    }

    pub fn get_data(&self) -> f32 {
        let v = self.0.as_ref();
        v.borrow().data
    }

    pub fn get_grad(&self) -> f32 {
        let v = self.0.as_ref();
        v.borrow().grad.unwrap()
    }

    pub fn get_option_grad(&self) -> Option<f32> {
        let v = self.0.as_ref();
        v.borrow().grad
    }

    pub fn get_creator(&self) -> Option<FuncBox> {
        self.0.clone().borrow().creator.clone()
    }

    pub fn get_gen(&self) -> u32 {
        self.0.clone().borrow().generation
    }

    pub fn set_grad(&self, grad: f32) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = Some(grad);
    }

    pub fn clear_grad(&self) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = None;
    }

    pub fn set_creator(&self, func: Rc<Function>) {
        let tmp = self.0.as_ref();
        let mut v = tmp.borrow_mut();
        v.generation = func.get_gen() + 1;
        v.creator = Some(FuncBox(func));
    }

    pub fn backward(&self) {
        if self.get_option_grad().is_none() {
            self.set_grad(1.);
        }

        let mut funcs = Vec::new();
        let mut seen_set = HashSet::new();

        add_func(&mut funcs, &mut seen_set, self.get_creator().unwrap());

        let mut funcs = vec![self.get_creator().unwrap()];
        while let Some(f) = funcs.pop() {
            let x = f.0.clone_input();
            let y = f.0.clone_output().iter().map(|y| y.get_grad()).collect();
            let gxs = f.0.backward(y);

            for (x, gx) in x.iter().zip(gxs.iter()) {
                if let Some(gx_old) = x.get_option_grad() {
                    x.set_grad(gx_old + gx)
                } else {
                    x.set_grad(*gx);
                }

                if let Some(x_creator) = x.get_creator() {
                    add_func(&mut funcs, &mut seen_set, x_creator)
                }
            }
        }
    }

    pub fn downgrade(self) -> WeakVBox {
        WeakVBox(Rc::downgrade(&self.0))
    }
}

#[derive(Clone)]
pub struct WeakVBox(Weak<RefCell<VariableData>>);

impl WeakVBox {
    fn get_grad(&self) -> f32 {
        let tmp = self.0.upgrade().unwrap();
        let v = tmp.as_ref();
        let x = v.borrow().grad.unwrap();
        x
    }
}

fn add_func(funcs: &mut Vec<FuncBox>, seen_set: &mut HashSet<FuncBox>, f: FuncBox) {
    if !seen_set.contains(&f) {
        funcs.push(f.clone());
        seen_set.insert(f);
        funcs.sort_by(|f, g| f.0.get_gen().cmp(&g.0.get_gen()))
    }
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        Variable::new($x)
    };
}

use crate::functions::{FType, FuncBox, Function};
mod operations;

use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    rc::{Rc, Weak},
};

#[allow(dead_code)]
struct Variable {
    name: Option<String>,
    data: f32,
    grad: Option<f32>,
    creator: Option<FuncBox>,
    generation: u32,
}

#[derive(Clone)]
pub struct VBox(Rc<RefCell<Variable>>);

impl std::fmt::Display for VBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut string = format!("Variable({}", self.get_data());
        match self.get_option_name() {
            None => {}
            Some(n) => string += &format!(", name: {n}"),
        }
        match self.get_option_grad() {
            None => {}
            Some(g) => string += &format!(", grad: {g}"),
        }
        string += ")";
        write!(f, "{}", string)
    }
}

impl VBox {
    pub fn new(data: f32) -> VBox {
        VBox(Rc::new(RefCell::new(Variable {
            name: None,
            data,
            grad: None,
            creator: None,
            generation: 0,
        })))
    }

    pub fn new_with_name(data: f32, name: &str) -> VBox {
        VBox(Rc::new(RefCell::new(Variable {
            name: Some(name.to_string()),
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

    fn get_option_name(&self) -> Option<String> {
        let v = self.0.as_ref();
        v.borrow().name.clone()
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
        self.backward_with_option(false);
    }

    pub fn backward_with_option(&self, retain_grad: bool) {
        if self.get_option_grad().is_none() {
            self.set_grad(1.);
        }

        let mut funcs = BinaryHeap::new();
        let mut seen_set = HashSet::new();

        let creator = self.get_creator().unwrap();
        funcs.push(creator.clone());
        seen_set.insert(creator);

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
                    if !seen_set.contains(&x_creator) {
                        funcs.push(x_creator.clone());
                        seen_set.insert(x_creator);
                    }
                }
            }

            if !retain_grad {
                for y in f.0.clone_output().iter() {
                    y.clear_grad()
                }
            }
        }
    }

    pub fn downgrade(self) -> WeakVBox {
        WeakVBox(Rc::downgrade(&self.0))
    }
}

#[derive(Clone)]
pub struct WeakVBox(Weak<RefCell<Variable>>);

impl WeakVBox {
    fn upgrade(&self) -> VBox {
        VBox(self.0.upgrade().unwrap())
    }

    fn get_grad(&self) -> f32 {
        let v = self.upgrade();
        v.get_grad()
    }

    fn clear_grad(&self) {
        let v = self.upgrade();
        v.clear_grad();
    }
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        $crate::variable::VBox::new($x as f32)
    };
}

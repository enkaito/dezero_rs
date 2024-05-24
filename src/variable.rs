use crate::array::Array;
use crate::functions::{FType, FuncBox, Function};
mod var_ops;

use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    rc::{Rc, Weak},
};

#[allow(dead_code)]
struct Variable {
    array: Array,
    grad: Option<Array>,
    creator: Option<FuncBox>,
    generation: u32,
}

#[derive(Clone)]
pub struct VBox(Rc<RefCell<Variable>>);

impl VBox {
    pub fn new(array: Array) -> VBox {
        VBox(Rc::new(RefCell::new(Variable {
            array,
            grad: None,
            creator: None,
            generation: 0,
        })))
    }

    pub fn get_array(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().array.clone()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        let v = self.0.as_ref();
        v.borrow().array.get_shape().clone()
    }

    pub fn get_grad(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().grad.clone().unwrap()
    }

    pub fn get_option_grad(&self) -> Option<Array> {
        let v = self.0.as_ref();
        v.borrow().grad.clone()
    }

    pub fn get_creator(&self) -> Option<FuncBox> {
        self.0.clone().borrow().creator.clone()
    }

    pub fn get_gen(&self) -> u32 {
        self.0.clone().borrow().generation
    }

    pub fn set_data(&self, data: Array) {
        let v = self.0.as_ref();
        v.borrow_mut().array.set_data(data);
    }

    pub fn set_grad(&self, grad: Array) {
        let v = self.0.as_ref();
        match &mut v.borrow_mut().grad {
            Some(grad_old) => grad_old.set_data(grad),
            x @ None => *x = Some(grad),
        };
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
            self.set_grad(Array::ones(&self.get_shape()));
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

            for (x, gx) in x.iter().zip(gxs.into_iter()) {
                if let Some(gx_old) = x.get_option_grad() {
                    x.set_grad(gx_old + &gx)
                } else {
                    x.set_grad(gx);
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

    fn get_grad(&self) -> Array {
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
        &$crate::variable::VBox::new($crate::array::Array::new(vec![$x as f32], vec![]))
    };
}

use crate::functions::FnBox;
use ndarray::{Array as A, IxDyn};
use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    hash::Hash,
    rc::Rc,
};

type Array = A<f32, IxDyn>;

use super::{Variable, WeakVBox};

#[derive(Clone)]
pub struct VBox(Rc<RefCell<Variable>>);

impl PartialEq for VBox {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for VBox {}

impl Hash for VBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize((Rc::as_ptr(&self.0) as *mut usize) as usize);
    }
}

impl VBox {
    pub fn new(array: Array) -> VBox {
        VBox(Rc::new(RefCell::new(Variable {
            array,
            grad: None,
            creator: None,
            generation: 0,
        })))
    }

    pub fn from_rc(rc: Rc<RefCell<Variable>>) -> VBox {
        VBox(rc)
    }

    pub fn get_array(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().array.clone()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        let v = self.0.as_ref();
        v.borrow().array.shape().to_vec()
    }

    pub fn get_grad(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().grad.clone().unwrap()
    }

    pub fn get_option_grad(&self) -> Option<Array> {
        let v = self.0.as_ref();
        v.borrow().grad.clone()
    }

    pub fn get_creator(&self) -> Option<FnBox> {
        self.0.clone().borrow().creator.clone()
    }

    pub fn get_gen(&self) -> u32 {
        self.0.clone().borrow().generation
    }

    pub fn set_array(&self, data: Array) {
        let v = self.0.as_ref();
        v.borrow_mut().array = data;
    }

    pub fn set_grad(&self, grad: Array) {
        let v = self.0.as_ref();
        match &mut v.borrow_mut().grad {
            Some(grad_old) => *grad_old = grad,
            x @ None => *x = Some(grad),
        };
    }

    pub fn clear_grad(&self) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = None;
    }

    pub fn set_creator(&self, func: FnBox) {
        let tmp = self.0.as_ref();
        let mut v = tmp.borrow_mut();
        v.generation = func.get_gen() + 1;
        v.creator = Some(func);
    }

    pub fn backward(&self) {
        self.backward_with_option(false);
    }

    pub fn backward_with_option(&self, retain_grad: bool) {
        if self.get_option_grad().is_none() {
            let shape = self.0.borrow().array.raw_dim();
            self.set_grad(Array::ones(shape));
        }

        let mut funcs = BinaryHeap::new();
        let mut seen_set = HashSet::new();

        let creator = self
            .get_creator()
            .expect("The creator of the variable is not set.\nMaybe backpropagation is disabled.");
        funcs.push(creator.clone());
        seen_set.insert(creator);

        while let Some(f) = funcs.pop() {
            let x = f.get_inputs();
            let gy = f.get_output().get_grad();
            let gxs = f.backward(gy);

            for (x, gx) in x.iter().zip(gxs.into_iter()) {
                if let Some(gx_old) = x.get_option_grad() {
                    x.set_grad(gx_old + gx)
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
                f.get_output().clear_grad();
            }
        }
    }

    pub fn downgrade(self) -> WeakVBox {
        WeakVBox::new(Rc::downgrade(&self.0))
    }
}

impl std::fmt::Display for VBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut string = format!("Variable(\n{}", self.get_array());
        match self.get_option_grad() {
            None => {}
            Some(g) => string += &format!(",\ngrad:\n{}", g.to_string()),
        }
        write!(f, "{}\n)", string)
    }
}

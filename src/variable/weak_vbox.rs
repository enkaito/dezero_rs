use super::{VBox, Variable};
use crate::Array;
use std::{cell::RefCell, rc::Weak};

#[derive(Clone)]
pub struct WeakVBox(Weak<RefCell<Variable>>);

impl WeakVBox {
    pub fn new(weak: Weak<RefCell<Variable>>) -> WeakVBox {
        WeakVBox(weak)
    }

    pub fn upgrade(&self) -> VBox {
        VBox::from_rc(self.0.upgrade().unwrap())
    }

    pub fn get_array(&self) -> Array {
        let v = self.upgrade();
        v.get_array()
    }

    pub fn get_grad(&self) -> Array {
        let v = self.upgrade();
        v.get_grad()
    }

    pub fn clear_grad(&self) {
        let v = self.upgrade();
        v.clear_grad();
    }
}

use std::collections::HashMap;

use crate::{layers::Model, Array, VBox};

pub trait Optimizer {
    fn update(&mut self) {
        let params = self.get_params();

        for param in params {
            self.update_one(param)
        }
    }
    fn get_params(&mut self) -> Vec<VBox>;
    fn update_one(&mut self, param: VBox);
}

pub struct SGD {
    lr: f32,
    target: Model,
}

impl SGD {
    pub fn new(lr: f32, target: Model) -> Self {
        SGD { lr, target }
    }
}

impl Optimizer for SGD {
    fn get_params(&mut self) -> Vec<VBox> {
        self.target.get_params()
    }
    fn update_one(&mut self, param: VBox) {
        param.set_array(param.get_array() - self.lr * param.get_grad())
    }
}

pub struct Momentum {
    lr: f32,
    momentum: f32,
    vs: HashMap<VBox, Array>,
    target: Model,
}

impl Momentum {
    pub fn new(lr: f32, momentum: f32, target: Model) -> Self {
        Momentum {
            lr,
            momentum,
            vs: HashMap::new(),
            target,
        }
    }
}

impl Optimizer for Momentum {
    fn get_params(&mut self) -> Vec<VBox> {
        self.target.get_params()
    }
    fn update_one(&mut self, param: VBox) {
        let v = self
            .vs
            .entry(param.clone())
            .or_insert_with(|| Array::zeros(&param.get_shape()));
        *v = &*v * self.momentum;
        *v = &*v - self.lr * param.get_grad();
        param.set_array(param.get_array() + &*v)
    }
}

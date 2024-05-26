use crate::functions as F;
use crate::variable::WeakVBox;
use crate::{Array, VBox};

pub trait Layer {
    fn call(&mut self, x: &VBox) -> VBox {
        let y = self.forward(x);
        self.set_io(x, &y);
        y
    }
    fn forward(&mut self, x: &VBox) -> VBox;
    fn clean_grads(&mut self);
    fn set_io(&mut self, input: &VBox, output: &VBox);
    fn get_params(&self) -> Vec<VBox>;
}

pub struct Linear {
    inputs: Option<WeakVBox>,
    outputs: Option<WeakVBox>,
    out_size: usize,
    w: Option<VBox>,
    b: Option<VBox>,
}

impl Linear {
    pub fn new(out_size: usize, bias: bool) -> Self {
        let b = if bias {
            Some(VBox::new(Array::zeros(&[out_size])))
        } else {
            None
        };

        Linear {
            inputs: None,
            outputs: None,
            out_size,
            w: None,
            b,
        }
    }

    fn init_w(&mut self, in_size: usize) {
        let w = VBox::new(Array::randn(
            &[in_size, self.out_size],
            0.,
            (in_size as f32).recip(),
        ));
        self.w = Some(w);
    }
}
impl Layer for Linear {
    fn forward(&mut self, x: &VBox) -> VBox {
        if self.w.is_none() {
            self.init_w(x.get_shape()[1]);
        }
        F::linear(&x, self.w.as_ref().unwrap(), self.b.as_ref())
    }
    fn clean_grads(&mut self) {
        self.w.as_ref().unwrap().clear_grad();
        if let Some(b) = &self.b {
            b.clear_grad();
        }
    }
    fn set_io(&mut self, input: &VBox, output: &VBox) {
        self.inputs = Some(input.clone().downgrade());
        self.outputs = Some(output.clone().downgrade());
    }
    fn get_params(&self) -> Vec<VBox> {
        let mut params = Vec::new();
        if let Some(w) = self.w.as_ref() {
            params.push(w.clone());
        }
        if let Some(b) = self.b.as_ref() {
            params.push(b.clone())
        }
        params
    }
}

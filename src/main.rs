use core::panic;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Rc<dyn Function>>,
}

#[derive(Clone, Debug)]
struct VBox(Rc<RefCell<Variable>>);

impl VBox {
    fn new(data: f32) -> VBox {
        VBox(Rc::new(RefCell::new(Variable::new(data))))
    }

    fn set_creator(&self, creator: Rc<dyn Function>) {
        let v = self.0.as_ref();
        v.borrow_mut().creator = Some(creator);
    }

    fn set_grad(&self, grad: f32) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = Some(grad);
    }
}

impl Variable {
    fn new(data: f32) -> Variable {
        Variable {
            data,
            grad: None,
            creator: None,
        }
    }
}

trait Function: Debug {
    fn call(&mut self, input: VBox) -> VBox {
        let x = input.0.borrow().data;
        let y = self.forward(x);
        let output = VBox::new(y);
        output.set_creator(self);
        self.set(&input, &output);
        output
    }

    fn new() -> Self
    where
        Self: Sized;
    fn input(&self) -> VBox;
    fn set(&mut self, input: &VBox, output: &VBox);
    fn forward(&self, x: f32) -> f32;
    fn backward(&self, gy: f32) -> f32;
}

#[derive(Debug)]
struct Square {
    input: Option<VBox>,
    output: Option<VBox>,
}

impl Function for Square {
    fn new() -> Self {
        Square {
            input: None,
            output: None,
        }
    }

    fn input(&self) -> VBox {
        let Some(input) = &self.input else {
            panic!("input is not set")
        };
        input.clone()
    }

    fn set(&mut self, input: &VBox, output: &VBox) {
        self.input = Some(input.clone());
        self.output = Some(output.clone());
    }

    fn forward(&self, x: f32) -> f32 {
        x.powi(2)
    }

    fn backward(&self, gy: f32) -> f32 {
        let input = self.input();
        let x = input.0.borrow().data;
        2. * x * gy
    }
}

#[derive(Debug)]
struct Exp {
    input: Option<VBox>,
    output: Option<VBox>,
}

impl Function for Exp {
    fn new() -> Self {
        Exp {
            input: None,
            output: None,
        }
    }

    fn input(&self) -> VBox {
        let Some(input) = &self.input else {
            panic!("input is not set")
        };
        input.clone()
    }

    fn set(&mut self, input: &VBox, output: &VBox) {
        self.input = Some(input.clone());
        self.output = Some(output.clone());
    }

    fn forward(&self, x: f32) -> f32 {
        x.exp()
    }

    fn backward(&self, gy: f32) -> f32 {
        let input = self.input();
        let x = input.0.borrow().data;
        x.exp() * gy
    }
}

fn main() {
    let mut A = Square::new();
    let mut B = Exp::new();
    let mut C = Square::new();

    let x = VBox::new(0.5);
    println!("{x:?}");
    let a = A.call(x.clone());
    println!("{a:?}");
    let b = B.call(a.clone());
    println!("{b:?}");
    let y = C.call(b.clone());
    println!("{y:?}");

    y.set_grad(1.);
    b.set_grad(C.backward(y.0.borrow().grad.unwrap()));
    a.set_grad(B.backward(b.0.borrow().grad.unwrap()));
    x.set_grad(A.backward(a.0.borrow().grad.unwrap()));

    println!("{x:?}");
}

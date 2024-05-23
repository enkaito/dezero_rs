use std::{
    cell::RefCell,
    fmt::Debug,
    rc::{Rc, Weak},
};

#[derive(Debug, Clone)]
struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Function>,
}

#[derive(Clone, Debug)]
struct VBox(Rc<RefCell<Variable>>);

impl VBox {
    fn new(data: f32) -> VBox {
        VBox(Rc::new(RefCell::new(Variable::new(data))))
    }

    fn get_data(&self) -> f32 {
        let v = self.0.as_ref();
        v.borrow().data
    }

    fn get_grad(&self) -> f32 {
        let v = self.0.as_ref();
        v.borrow().grad.unwrap()
    }

    fn get_option_grad(&self) -> Option<f32> {
        let v = self.0.as_ref();
        v.borrow().grad
    }

    fn get_creator(&self) -> Option<Function> {
        self.0.clone().borrow().creator.clone()
    }

    fn set_grad(&self, grad: f32) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = Some(grad);
    }

    fn set_creator(&self, creator: Function) {
        let v = self.0.as_ref();
        v.borrow_mut().creator = Some(creator);
    }

    fn backward(&self) {
        if self.get_option_grad().is_none() {
            self.set_grad(1.);
        }

        let mut funcs = vec![self.get_creator().unwrap()];
        while let Some(f) = funcs.pop() {
            let x = f.input.clone().unwrap();
            let y = f.output.clone().unwrap();

            x.set_grad(f.backward(y.get_grad()));

            if let Some(x_creator) = x.get_creator() {
                funcs.push(x_creator)
            };
        }
    }

    fn downgrade(self) -> WeakVBox {
        WeakVBox(Rc::downgrade(&self.0))
    }
}

#[derive(Debug, Clone)]
struct WeakVBox(Weak<RefCell<Variable>>);

impl WeakVBox {
    fn get_grad(&self) -> f32 {
        let tmp = self.0.upgrade().unwrap();
        let v = tmp.as_ref();
        let x = v.borrow().grad.unwrap();
        x
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

#[derive(Debug, Clone)]
enum FType {
    Square,
    Exp,
}

#[derive(Debug, Clone)]
struct Function {
    input: Option<VBox>,
    output: Option<WeakVBox>,
    ftype: FType,
}

impl Function {
    fn new(ftype: FType) -> Function {
        Function {
            input: None,
            output: None,
            ftype: ftype,
        }
    }

    fn call(mut self, input: &VBox) -> VBox {
        let x = input.get_data();
        let y = self.forward(x);
        let output = VBox::new(y);
        self.input = Some(input.clone());
        self.output = Some(output.clone().downgrade());
        output.set_creator(self);
        output
    }

    fn forward(&self, x: f32) -> f32 {
        match &self.ftype {
            FType::Square => x.powi(2),
            FType::Exp => x.exp(),
        }
    }

    fn backward(&self, gy: f32) -> f32 {
        let x = self.input.as_ref().unwrap().get_data();
        let pdv = match &self.ftype {
            FType::Square => 2. * x,
            FType::Exp => x.exp(),
        };
        pdv * gy
    }
}

macro_rules! square {
    ($x: expr) => {{
        let func = Function::new(FType::Square);
        func.call(&$x)
    }};
}

macro_rules! exp {
    ($x: expr) => {{
        let func = Function::new(FType::Exp);
        func.call(&$x)
    }};
}

fn main() {
    let x = VBox::new(0.5);

    let y = square!(exp!(square!(x)));

    y.backward();
    println!("{}", x.get_grad())
}

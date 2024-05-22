struct Variable {
    data: f32,
    grad: Option<f32>,
    creator: Option<Box<dyn Function>>,
}

impl Variable {
    fn new(data: f32) -> Variable {
        Variable {
            data,
            grad: None,
            creator: None,
        }
    }

    fn set_creator(&mut self, func: impl Function + 'static) {
        self.creator = Some(Box::new(func));
    }

    fn backward(&mut self) {
        if let Some(f) = self.creator {
            let x = f.input;
            x.grad = f.backward(self.grad);
            x.backward();
        };
    }
}

trait Function {
    fn call(&self, input: Variable) -> Variable;
    fn forward(&self, x: Variable) -> Variable;
    fn backward(&self, gy: Variable) -> Variable;
}

fn main() {
    println!("Hello, world!");
}

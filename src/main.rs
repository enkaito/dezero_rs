#[allow(unused_imports)]
use dezero::{array1, functions as F, var, Array};

fn main() {
    let x = var!(array1!([0.2, -0.4, 0.3, 0.5, 1.3, -3.2, 2.1, 0.3]).reshape(&[4, 2]));
    let t = var!(array1!([0, 1, 1, 0, 0, 1, 1, 0]).reshape(&[4, 2]));
    let y = &F::softmax(x, 1);
    let loss = &F::cross_entropy_loss(y, t);
    loss.backward();

    println!("{}", x);
    println!("{}", t);
    println!("{}", y);
}

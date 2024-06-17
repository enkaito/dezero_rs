use dezero::variable::VBox;
use ndarray::{Array, IxDyn};

fn main() {
    let x = &VBox::new(Array::from_elem(IxDyn(&[]), 0.5));
    let y = &VBox::new(Array::from_elem(IxDyn(&[]), 1.2));
    let z = x + y;

    z.backward();
    
    println!("{}", z);
    println!("{}", y);
    println!("{}", x);
}

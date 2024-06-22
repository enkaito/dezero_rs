use dezero::{array, variable::VBox};

fn main() {
    let x = &VBox::new(array!(vec![1., 2., 3., 4.], &[2, 2]));
    let y = &VBox::new(array!(vec![1., 2., 3., 4.], &[2, 2]));
    let z = x + y;

    z.backward();

    println!("{}", z);
    println!("{}", y);
    println!("{}", x);
}

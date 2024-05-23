use dezero::functions::{FType, Function};
use dezero::variable::VBox;
use dezero::{add, exp, square};

fn main() {
    let x = VBox::new(2.);
    let y = VBox::new(3.);
    println!("x: {}\ny: {}", x, y);

    let z = add!(add!(x, x), x);

    z.backward();
    println!("{}", z);
    println!("{}", x);
    println!("{}", y);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn square_backward_test() {
        let x = VBox::new(3.);
        let y = square!(x);
        y.backward();
        assert_eq!(x.get_grad(), 6.);
    }

    #[test]
    fn add_test() {
        let x0 = VBox::new(2.);
        let x1 = VBox::new(3.);
        let y = add!(x0, x1);
        assert_eq!(y.get_data(), 5.);
    }

    #[test]
    fn square_add_test() {
        let x = VBox::new(2.);
        let y = VBox::new(3.);
        let z = add!(square!(x), square!(y));
        z.backward();
        assert_eq!(z.get_data(), 13.);
        assert_eq!(x.get_grad(), 4.);
        assert_eq!(y.get_grad(), 6.);
    }

    #[test]
    fn add_backward_test() {
        let x = VBox::new(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);
    }

    #[test]
    fn clear_grad_test() {
        let x = VBox::new(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);

        x.clear_grad();
        let y = add!(add!(x, x), x);
        y.backward();
        assert_eq!(x.get_grad(), 3.);
    }
}

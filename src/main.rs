use dezero::functions::{FType, Function};
use dezero::variable::Variable;
use dezero::{add, exp, square, var};

fn main() {
    let x = var!(2.);
    let a = square!(x);
    let y = add!(square!(a), square!(a));
    y.backward();

    println!("{}", y.get_data());
    println!("{}", x.get_grad());
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn square_backward_test() {
        let x = Variable::new(3.);
        let y = square!(x);
        y.backward();
        assert_eq!(x.get_grad(), 6.);
    }

    #[test]
    fn add_test() {
        let x0 = Variable::new(2.);
        let x1 = Variable::new(3.);
        let y = add!(x0, x1);
        assert_eq!(y.get_data(), 5.);
    }

    #[test]
    fn square_add_test() {
        let x = Variable::new(2.);
        let y = Variable::new(3.);
        let z = add!(square!(x), square!(y));
        z.backward();
        assert_eq!(z.get_data(), 13.);
        assert_eq!(x.get_grad(), 4.);
        assert_eq!(y.get_grad(), 6.);
    }

    #[test]
    fn add_backward_test() {
        let x = Variable::new(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);
    }

    #[test]
    fn clear_grad_test() {
        let x = Variable::new(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);

        x.clear_grad();
        let y = add!(add!(x, x), x);
        y.backward();
        assert_eq!(x.get_grad(), 3.);
    }
}

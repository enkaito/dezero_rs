use dezero::VBox;
#[allow(unused_imports)]
use dezero::{add, exp, square, var};

fn main() {
    let matyas = |x: &VBox, y: &VBox| 0.26 * (x.pow(2.) + y.pow(2.)) - 0.48 * x * y;

    let x = &var!(1.);
    let y = &var!(1.);
    let z = matyas(x, y);
    z.backward();

    println!("{}", x);
    println!("{}", y);
    println!("{}", z);

    let x = &var!(1.);
    let z = x.pow(2.);
    z.backward();
    println!("{}", x);
}

#[cfg(test)]
mod test {
    use dezero::{add, square, var, variable::VBox};

    #[test]
    fn square_backward_test() {
        let x = var!(3.);
        let y = square!(x);
        y.backward();
        assert_eq!(x.get_grad(), 6.);
    }

    #[test]
    fn add_test() {
        let x0 = var!(2.);
        let x1 = var!(3.);
        let y = add!(x0, x1);
        assert_eq!(y.get_data(), 5.);
    }

    #[test]
    fn square_add_test() {
        let x = var!(2.);
        let y = var!(3.);
        let z = add!(square!(x), square!(y));
        z.backward();
        assert_eq!(z.get_data(), 13.);
        assert_eq!(x.get_grad(), 4.);
        assert_eq!(y.get_grad(), 6.);
    }

    #[test]
    fn add_backward_test() {
        let x = var!(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);
    }

    #[test]
    fn clear_grad_test() {
        let x = var!(3.);
        let y = add!(x, x);
        y.backward();
        assert_eq!(x.get_grad(), 2.);

        x.clear_grad();
        let y = add!(add!(x, x), x);
        y.backward();
        assert_eq!(x.get_grad(), 3.);
    }

    #[test]
    fn complex_graph_test() {
        let x = var!(2.);
        let a = square!(x);
        let y = add!(square!(a), square!(a));
        y.backward();

        assert_eq!(y.get_data(), 32.);
        assert_eq!(x.get_grad(), 64.);
    }

    #[test]
    fn grad_drop_test() {
        let x0 = var!(1.);
        let x1 = var!(1.);

        let t = add!(x0, x1);
        let y = add!(x0, t);

        y.backward();

        assert_eq!(y.get_option_grad(), None);
        assert_eq!(t.get_option_grad(), None);
        assert_eq!(x0.get_option_grad(), Some(2.));
        assert_eq!(x1.get_option_grad(), Some(1.));
    }

    #[test]
    fn overload_test() {
        let a = &var!(3.);
        let b = &var!(2.);
        let c = &var!(1.);

        let y = a * b + c;
        y.backward();

        assert_eq!(y.get_data(), 7.);
        assert_eq!(a.get_grad(), 2.);
        assert_eq!(b.get_grad(), 3.);
    }

    #[test]
    fn test_sphere() {
        let x = var!(1.);
        let y = var!(1.);
        let z = x.pow(2.) + y.pow(2.);

        z.backward();

        assert_eq!(x.get_grad(), 2.);
        assert_eq!(y.get_grad(), 2.);
    }

    #[test]
    fn test_matyas() {
        let matyas = |x: &VBox, y: &VBox| 0.26 * (x.pow(2.) + y.pow(2.)) - 0.48 * x * y;

        let x = &var!(1.);
        let y = &var!(1.);
        let z = matyas(x, y);
        z.backward();

        assert!((x.get_grad() - 0.04) < 1e-8);
        assert!((y.get_grad() - 0.04) < 1e-8);
    }

    #[test]
    fn test_goldstein_prince() {
        let gp = |x: &VBox, y: &VBox| {
            (1 + (x + y + 1).pow(2.)
                * (19 - 14 * x + 3 * x.pow(2.) - 14 * y + 6 * x * y + 3 * y.pow(2.)))
                * (30
                    + (2 * x - 3 * y).pow(2.)
                        * (18 - 32 * x + 12 * x.pow(2.) + 48 * y - 36 * x * y + 27 * y.pow(2.)))
        };

        let x = &var!(1);
        let y = &var!(1);
        let z = gp(x, y);
        z.backward();

        assert!((x.get_grad() + 5376.) < 1e-8);
        assert!((y.get_grad() - 8064.) < 1e-8);
    }
}

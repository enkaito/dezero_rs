#[allow(unused_imports)]
use dezero::{array0, array1, array2, array_with_shape, eval, var, Array, VBox};

fn main() {
    eval! {{
        let x = VBox::new(array_with_shape!(1..=6, [2, 3]));
        let y = x.reshape(vec![6]);
        y.backward();
        println!("{}", x);
        println!("{}", y);
    }}
}

#[cfg(test)]
mod test {
    use dezero::{array0, var, VBox};

    macro_rules! square {
        ($x: expr) => {
            $x.clone() * $x.clone()
        };
    }

    #[test]
    fn square_backward_test() {
        let x = var!(3.);
        let y = square!(x);
        y.backward();
        assert_eq!(x.get_grad(), array0!(6));
    }

    #[test]
    fn add_test() {
        let x0 = var!(2.);
        let x1 = var!(3.);
        let y = x0 + x1;
        assert_eq!(y.get_array(), array0!(5));
    }

    #[test]
    fn square_add_test() {
        let x = var!(2.);
        let y = var!(3.);
        let z = square!(x) + square!(y);
        z.backward();
        assert_eq!(z.get_array(), array0!(13));
        assert_eq!(x.get_grad(), array0!(4));
        assert_eq!(y.get_grad(), array0!(6));
    }

    #[test]
    fn add_backward_test() {
        let x = var!(3.);
        let y = x + x;
        y.backward();
        assert_eq!(x.get_grad(), array0!(2));
    }

    #[test]
    fn clear_grad_test() {
        let x = var!(3.);
        let y = x + x;
        y.backward();
        assert_eq!(x.get_grad(), array0!(2.));

        x.clear_grad();
        let y = x + x + x;
        y.backward();
        assert_eq!(x.get_grad(), array0!(3.));
    }

    #[test]
    fn complex_graph_test() {
        let x = var!(2.);
        let a = square!(x);
        let y = square!(a) + square!(a);
        y.backward();

        assert_eq!(y.get_array(), array0!(32.));
        assert_eq!(x.get_grad(), array0!(64.));
    }

    #[test]
    fn grad_drop_test() {
        let x0 = var!(1.);
        let x1 = var!(1.);

        let t = x0 + x1;
        let y = x0 + &t;

        y.backward();

        assert_eq!(y.get_option_grad(), None);
        assert_eq!(t.get_option_grad(), None);
        assert_eq!(x0.get_option_grad(), Some(array0!(2.)));
        assert_eq!(x1.get_option_grad(), Some(array0!(1.)));
    }

    #[test]
    fn overload_test() {
        let a = var!(3.);
        let b = var!(2.);
        let c = var!(1.);

        let y = a * b + c;
        y.backward();

        assert_eq!(y.get_array(), array0!(7.));
        assert_eq!(a.get_grad(), array0!(2.));
        assert_eq!(b.get_grad(), array0!(3.));
    }

    #[test]
    fn test_sphere() {
        let x = var!(1.);
        let y = var!(1.);
        let z = x.powi(2) + y.powi(2);

        z.backward();

        assert_eq!(x.get_grad(), array0!(2.));
        assert_eq!(y.get_grad(), array0!(2.));
    }

    #[test]
    fn test_matyas() {
        let matyas = |x: &VBox, y: &VBox| 0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y;

        let x = &var!(1.);
        let y = &var!(1.);
        let z = matyas(x, y);
        z.backward();

        assert!((x.get_grad() - 0.04).sum() < 1e-8);
        assert!((y.get_grad() - 0.04).sum() < 1e-8);
    }

    #[test]
    fn test_goldstein_prince() {
        let gp = |x: &VBox, y: &VBox| {
            (1 + (x + y + 1).powi(2)
                * (19 - 14 * x + 3 * x.powi(2) - 14 * y + 6 * x * y + 3 * y.powi(2)))
                * (30
                    + (2 * x - 3 * y).powi(2)
                        * (18 - 32 * x + 12 * x.powi(2) + 48 * y - 36 * x * y + 27 * y.powi(2)))
        };

        let x = &var!(1);
        let y = &var!(1);
        let z = gp(x, y);
        z.backward();

        assert!(x.get_grad().all_close(&array0!(-5376.), 1e-8));
        assert!(y.get_grad().all_close(&array0!(8064), 1e-8));
    }

    #[test]
    fn rosenbrock_opt_test() {
        let rosenbrock = |x0: &VBox, x1: &VBox| 100 * (x1 - x0.powi(2)).powi(2) + (x0 - 1).powi(2);

        let x0 = var!(0);
        let x1 = var!(2);

        let lr = 0.001;
        let max_iter = 100;

        for _ in 0..max_iter {
            println!("{}, {}", x0, x1);

            let y = rosenbrock(x0, x1);

            println!("{}", y);

            x0.clear_grad();
            x1.clear_grad();
            y.backward();

            x0.set_data(x0.get_array() - lr * x0.get_grad());
            x1.set_data(x0.get_array() - lr * x0.get_grad());
        }

        assert!(x0.get_grad().all_close(&array0!(1), 1e-8));
        assert!(x1.get_grad().all_close(&array0!(1), 1e-8));
    }
}

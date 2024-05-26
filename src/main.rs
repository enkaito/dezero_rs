use std::sync::Arc;

#[allow(unused_imports)]
use dezero::{array0, array1, array2, array_with_shape, eval, scaler, var, Array, VBox};

fn main() {
    // let x = var!(Array::rand(&[100, 1]));
    // let y = 5 + 2 * x + var!(Array::rand(&[100, 1]));

    // let w = var!(Array::zeros(&[1, 1]));
    // let b = var!(Array::zeros(&[]));

    // let pred = |x: &VBox| x.dot(&w) + b;
    // let mean_squared_error = |x: &VBox, y: &VBox| (x - y).powi(2).sum();

    // let lr = 0.1;
    // let iters = 100;

    // for _ in 0..iters {
    //     let y_pred = pred(x);
    //     let loss = mean_squared_error(&y, &y_pred);

    //     w.clear_grad();
    //     b.clear_grad();
    //     loss.backward();

    //     w.set_array(w.get_array() - lr * w.get_grad());
    //     b.set_array(b.get_array() - lr * b.get_grad());

    //     println!("w:\n{}", w);
    //     println!("b:\n{}", b);
    //     println!("loss:\n{}\n", loss);
    // }

    let x = array1!(0..8).reshape(&[2, 2, 2]);
    let y = x.sum_to(&[]);
    println!("{x}");
    println!();
    println!("{y}");
}

#[cfg(test)]
mod test {
    #[allow(unused_imports)]
    use dezero::{array0, array1, array2, array_with_shape, scaler, var, VBox};

    macro_rules! square {
        ($x: expr) => {
            $x.clone() * $x.clone()
        };
    }

    #[test]
    fn square_backward_test() {
        let x = scaler!(3.);
        let y = square!(x);
        y.backward();
        assert_eq!(x.get_grad(), array0!(6));
    }

    #[test]
    fn add_test() {
        let x0 = scaler!(2.);
        let x1 = scaler!(3.);
        let y = x0 + x1;
        assert_eq!(y.get_array(), array0!(5));
    }

    #[test]
    fn square_add_test() {
        let x = scaler!(2.);
        let y = scaler!(3.);
        let z = square!(x) + square!(y);
        z.backward();
        assert_eq!(z.get_array(), array0!(13));
        assert_eq!(x.get_grad(), array0!(4));
        assert_eq!(y.get_grad(), array0!(6));
    }

    #[test]
    fn add_backward_test() {
        let x = scaler!(3.);
        let y = x + x;
        y.backward();
        assert_eq!(x.get_grad(), array0!(2));
    }

    #[test]
    fn clear_grad_test() {
        let x = scaler!(3.);
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
        let x = scaler!(2.);
        let a = square!(x);
        let y = square!(a) + square!(a);
        y.backward();

        assert_eq!(y.get_array(), array0!(32.));
        assert_eq!(x.get_grad(), array0!(64.));
    }

    #[test]
    fn grad_drop_test() {
        let x0 = scaler!(1.);
        let x1 = scaler!(1.);

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
        let a = scaler!(3.);
        let b = scaler!(2.);
        let c = scaler!(1.);

        let y = a * b + c;
        y.backward();

        assert_eq!(y.get_array(), array0!(7.));
        assert_eq!(a.get_grad(), array0!(2.));
        assert_eq!(b.get_grad(), array0!(3.));
    }

    #[test]
    fn test_sphere() {
        let x = scaler!(1.);
        let y = scaler!(1.);
        let z = x.powi(2) + y.powi(2);

        z.backward();

        assert_eq!(x.get_grad(), array0!(2.));
        assert_eq!(y.get_grad(), array0!(2.));
    }

    #[test]
    fn test_matyas() {
        let matyas = |x: &VBox, y: &VBox| 0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y;

        let x = &scaler!(1.);
        let y = &scaler!(1.);
        let z = matyas(x, y);
        z.backward();

        assert!(x.get_grad().all_close(&array0!(0.04), 1e-8));
        assert!(y.get_grad().all_close(&array0!(0.04), 1e-8));
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

        let x = &scaler!(1);
        let y = &scaler!(1);
        let z = gp(x, y);
        z.backward();

        assert!(x.get_grad().all_close(&array0!(-5376.), 1e-8));
        assert!(y.get_grad().all_close(&array0!(8064), 1e-8));
    }

    #[test]
    fn rosenbrock_opt_test() {
        let rosenbrock = |x0: &VBox, x1: &VBox| 100 * (x1 - x0.powi(2)).powi(2) + (x0 - 1).powi(2);

        let x0 = scaler!(0);
        let x1 = scaler!(2);

        let lr = 0.001;
        let max_iter = 100;

        for _ in 0..max_iter {
            println!("{}, {}", x0, x1);

            let y = rosenbrock(x0, x1);

            println!("{}", y);

            x0.clear_grad();
            x1.clear_grad();
            y.backward();

            x0.set_array(x0.get_array() - lr * x0.get_grad());
            x1.set_array(x0.get_array() - lr * x0.get_grad());
        }

        assert!(x0.get_grad().all_close(&array0!(1), 1e-8));
        assert!(x1.get_grad().all_close(&array0!(1), 1e-8));
    }

    #[test]
    fn reshape_test() {
        let x = var!(array_with_shape!(0..6, [2, 3]));
        let y = x.reshape(vec![6]);
        y.backward();
        assert_eq!(x.get_grad(), array_with_shape!([1; 6], [2, 3]));
        assert_eq!(y.get_array(), array_with_shape!(0..6, [6]))
    }

    #[test]
    fn transpose_test() {
        let x = var!(array2!([[1, 2, 3], [4, 5, 6]]));
        let y = x.transpose();
        y.backward();
        assert_eq!(x.get_grad(), array_with_shape!([1; 6], [2, 3]));
        assert_eq!(y.get_array(), array_with_shape!([1, 4, 2, 5, 3, 6], [3, 2]))
    }
}

use dezero::{add, exp, square, var};

fn main() {
    let x = add!(var!(2.), var!(3.), false);
}

#[cfg(test)]
mod test {
    use dezero::{add, exp, square, var};

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
}

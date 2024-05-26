use dezero::functions as F;
#[allow(unused_imports)]
use dezero::{array0, array1, array2, array_with_shape, eval, scaler, var, Array, VBox};
use std::f32::consts::PI;

fn main() {
    let x = (array1!(0..100) / 100.).reshape(&[100, 1]);
    let y = (2. * PI * &x).sin() + Array::rand(&[100, 1]) - 0.5;
    let x = var!(x.reshape(&[100, 1]));
    let y = var!(y);

    let w1 = var!((Array::rand(&[1, 10]) - 0.5) * 0.01);
    let b1 = var!(Array::zeros(&[10]));
    let w2 = var!((Array::rand(&[10, 1]) - 0.5) * 0.01);
    let b2 = var!(Array::zeros(&[1]));

    let pred = |x: &VBox| {
        let y = F::linear(x, w1, Some(b1));
        let y = F::sigmoid(&y);
        let y = F::linear(&y, w2, Some(b2));
        y
    };

    let lr = 0.01;
    let iters = 10000;

    for i in 0..iters {
        let y_pred = &pred(x);
        let loss = &F::mean_squared_error(y, y_pred);

        w1.clear_grad();
        b1.clear_grad();
        w2.clear_grad();
        b2.clear_grad();
        loss.backward();

        w1.set_array(w1.get_array() - lr * w1.get_grad());
        b1.set_array(b1.get_array() - lr * b1.get_grad());
        w2.set_array(w2.get_array() - lr * w2.get_grad());
        b2.set_array(b2.get_array() - lr * b2.get_grad());

        if i % 1000 == 0 {
            println!("{loss}");
        }
    }
}

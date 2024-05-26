use dezero::functions as F;
use dezero::layers as L;
use dezero::layers::Layer;
#[allow(unused_imports)]
use dezero::{array0, array1, array2, array_with_shape, eval, scaler, var, Array, VBox};
use std::f32::consts::PI;

fn main() {
    let x = (array1!(0..100) / 100.).reshape(&[100, 1]);
    let y = (2. * PI * &x).sin() + Array::rand(&[100, 1]) - 0.5;
    let x = var!(x.reshape(&[100, 1]));
    let y = var!(y);

    let mut l = L::MLP::new(&[10, 1], Box::new(F::sigmoid));

    let lr = 0.005;
    let iters = 10000;

    for i in 0..iters {
        let y_pred = &l.call(x);
        let loss = &F::mean_squared_error(y, y_pred);

        l.clear_grads();
        loss.backward();

        for p in l.get_params() {
            p.set_array(p.get_array() - lr * p.get_grad());
        }

        if i % 1000 == 0 {
            println!("{loss}");
        }
    }
}

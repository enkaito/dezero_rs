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

    let mut l1 = L::Linear::new(10, true);
    let mut l2 = L::Linear::new(1, true);

    let pred = |x: &VBox, l1: &mut dyn Layer, l2: &mut dyn Layer| {
        let y = l1.call(x);
        let y = F::sigmoid(&y);
        let y = l2.call(&y);
        y
    };

    let lr = 0.005;
    let iters = 10000;

    for i in 0..iters {
        let y_pred = &pred(x, &mut l1, &mut l2);
        let loss = &F::mean_squared_error(y, y_pred);

        l1.clean_grads();
        l2.clean_grads();
        loss.backward();

        for l in &[&l1, &l2] {
            for p in l.get_params() {
                p.set_array(p.get_array() - lr * p.get_grad());
            }
        }

        if i % 1000 == 0 {
            println!("{loss}");
        }
    }
}

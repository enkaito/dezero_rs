extern crate dezero;

use dezero::{array::Array, array0, array1};

#[test]
fn matmul1() {
    let x = array1!(0..16).reshape(&[8, 2]);
    let y = array1!(0..4).reshape(&[2, 2]);

    assert_eq!(
        x.matmul(&y),
        array1!([2, 3, 6, 11, 10, 19, 14, 27, 18, 35, 22, 43, 26, 51, 30, 59]).reshape(&[8, 2])
    );
}

#[test]
fn matmul2() {
    let x = array1!(0..16).reshape(&[2, 4, 2]);
    let y = array1!(0..4).reshape(&[2, 2]);

    assert_eq!(
        x.matmul(&y),
        array1!([2, 3, 6, 11, 10, 19, 14, 27, 18, 35, 22, 43, 26, 51, 30, 59]).reshape(&[2, 4, 2])
    );
}

#[test]
fn matmul3() {
    let x = array1!([1, 2, 3, 4, 5]);
    assert_eq!(x.matmul(&x), array0!(55));
}

#[test]
fn matmul4() {
    let a = Array::ones(&[9, 5, 7, 4]);
    let c = Array::ones(&[9, 5, 4, 3]);

    assert_eq!(a.matmul(&c).get_shape(), &[9, 5, 7, 3])
}

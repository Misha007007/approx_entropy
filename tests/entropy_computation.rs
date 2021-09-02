use approx_entropy::prelude::*;
use float_eq::assert_float_eq;
use rand::rngs::ThreadRng;
use test_case::test_case;

#[test_case([1, 2, 3, 4, 5, 6], -0.144405662; "increasing")]
#[test_case(vec!['a', 'b', 'c', 'd', 'd', 'e', 'e', 'e'], -0.144405662; "uniform_3")]
fn entropy<T>(source: T, expected: f64)
where
    Estimator<Bootstrap<ThreadRng>>: From<T>,
{
    let num_groups = 3;
    let degree = 2;
    let rng = rng(1);
    let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();
    let mut estimator = Estimator::from(source).set_sampling_method(bootstrap);

    assert_float_eq!(estimator.entropy(), expected, abs <= 1e-6);
}

fn rng(seed: u64) -> impl rand::RngCore {
    const INC: u64 = 11634580027462260723;
    rand_pcg::Pcg32::new(seed, INC)
}

use core::hash::Hash;
use nalgebra::DMatrix;
use rand::rngs::ThreadRng;

use crate::{Bootstrap, SamplingMethod};

mod naive;

pub use naive::NaiveEstimator;

#[derive(Debug, PartialEq)]
pub struct Estimator<M> {
    sample_size: usize,
    num_groups: usize,
    degree: usize,
    // sample counting reps
    sample: Vec<usize>,
    pub sampling_method: M,
}

impl<M> Estimator<M>
where
    M: SamplingMethod,
{
    pub fn new(
        unnorm_distr: &[usize],
        num_groups: usize,
        degree: usize,
        sampling_method: M,
    ) -> Self {
        let sample_size = unnorm_distr.iter().sum();
        Estimator {
            sample_size,
            num_groups,
            degree,
            sample: unnorm_distr.to_vec(),
            sampling_method,
        }
    }

    pub fn entropy(&mut self) -> f64 {
        let y = self.sampling_method.sample_entropy();

        let x = {
            let mut vec_x = Vec::new();
            let size_subsamples = self.sampling_method.size_subsamples();
            let simple_x = DMatrix::from_fn(size_subsamples.len(), self.degree + 1, |r, c| {
                size_subsamples[r] as f64 / (size_subsamples[r].pow(c as u32)) as f64
            });

            for col in 0..self.degree + 1 {
                for row in 0..size_subsamples.len() {
                    vec_x.push(simple_x[(row, col)]);
                }
            }
            DMatrix::<f64>::from_vec(self.sampling_method.total_samples(), self.degree + 1, vec_x)
        };

        // Least squares for `x ? = b`
        let x_t = x.transpose();
        let b = x_t.clone() * y;
        let a = x_t * x;
        a.lu().solve(&b).unwrap()[0]
    }
}

impl<T> From<&[T]> for Estimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    fn from(samples: &[T]) -> Self {
        let unnorm_distr = crate::count_dup(&samples);
        let num_groups = 3;
        let degree = 2;
        let sampling_method = Bootstrap::new(rand::thread_rng());
        Estimator::new(&unnorm_distr, num_groups, degree, sampling_method)
    }
}

impl<T> From<Vec<T>> for Estimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    fn from(samples: Vec<T>) -> Self {
        <Estimator<Bootstrap<ThreadRng>> as From<&[T]>>::from(&samples)
    }
}

impl<T> From<&Vec<T>> for Estimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    fn from(samples: &Vec<T>) -> Self {
        <Estimator<Bootstrap<ThreadRng>> as From<&[T]>>::from(samples)
    }
}

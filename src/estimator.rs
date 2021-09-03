use core::hash::Hash;
use nalgebra::{DMatrix, DVector};
use rand::rngs::ThreadRng;
use thiserror::Error;

use crate::{Bootstrap, SamplingMethod};

mod direct;
mod naive;

pub use direct::DirectEstimator;
pub use naive::NaiveEstimator;

const DEFAULT_NUM_GROUPS: usize = 3;
const DEFAULT_DEGREE: usize = 2;

/// Entropy estimator
///
/// This is a wrapper around a concrete sampling method.
///
/// # Examples
///
/// Quick estimation from an unormalized distribution.
/// ```
/// # use approx_entropy::Estimator;
/// let unnorm_distr = [1, 2, 3, 4, 5, 6];
/// let mut estimator = Estimator::from(unnorm_distr);
/// println!("Entropy estimation: {:?}", estimator.entropy()); // Random result
/// ```
///
/// Quick estimation from a sample.
/// ```
/// # use approx_entropy::Estimator;
/// let samples = vec![1, 2, 3, 1, 1, 2, 2, 1, 3]; // samples from a random variable
/// let mut estimator = Estimator::from(samples);
/// println!("Entropy estimation: {:?}", estimator.entropy()); // Random result
/// ```
#[derive(Debug, PartialEq)]
pub struct Estimator<M> {
    sampling_method: M,
}

#[derive(Error, Debug)]
#[error("Failed to estimate entropy because of numerical instability.")]
pub struct FittingError;

/// # Basic methods
impl<M> Estimator<M>
where
    M: SamplingMethod,
{
    /// Constructs a new `Estimator`.
    ///
    /// # Remarks
    ///
    /// The trait `From<M>` is also implemented for convenience.
    pub fn new(sampling_method: M) -> Self {
        Estimator { sampling_method }
    }
    /// Estimates the entropy of the underlying distribution,
    /// known only through the empirical unnormalized distribution.
    ///
    /// # Errors
    ///
    /// If there are numerical instabilities.
    pub fn entropy(&mut self) -> Result<f64, FittingError> {
        let (size_subsamples_dup, scaled_naive_entropies): (Vec<_>, Vec<_>) = self
            .sampling_method
            .naive_entropies()
            .into_iter()
            .map(|(size, value)| (size, value * size as f64))
            .unzip();

        // Fitting a polynomial
        let y = DVector::from_vec(scaled_naive_entropies);
        let x = DMatrix::<f64>::from_fn(
            self.sampling_method.total_samples(),
            self.sampling_method.degree() + 1,
            |r, c| (size_subsamples_dup[r] as f64).powi(1 - c as i32),
        );

        // Least squares for `x ? = y`
        let x_t = x.transpose();
        let b = x_t.clone() * y;
        let a = x_t * x;

        match a.lu().solve(&b) {
            Some(polynomial) => Ok(polynomial[0]),
            None => Err(FittingError),
        }
    }
}

/// # Getters
///
/// Get the underlying sampling method.
impl<M> Estimator<M>
where
    M: SamplingMethod,
{
    /// Returns the underlying sampling method.
    pub fn sampling_method(&self) -> &M {
        &self.sampling_method
    }

    /// Returns the underlying sampling method.
    pub fn sampling_method_mut(&mut self) -> &mut M {
        &mut self.sampling_method
    }
}

/// # Transformations
impl<M> Estimator<M> {
    pub fn set_sampling_method<M2>(self, other: M2) -> Estimator<M2>
    where
        M2: SamplingMethod,
    {
        Estimator {
            sampling_method: other,
        }
    }
}

impl<M> From<M> for Estimator<M>
where
    M: SamplingMethod,
{
    fn from(sampling_method: M) -> Self {
        Self::new(sampling_method)
    }
}

impl<const N: usize> From<[usize; N]> for Estimator<Bootstrap<ThreadRng>> {
    /// Performs the conversion from an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `Estimator`,
    /// but be aware that default values are given to tunable parameters.
    fn from(unnorm_distr: [usize; N]) -> Self {
        let sampling_method = Bootstrap::new(
            &unnorm_distr,
            DEFAULT_NUM_GROUPS,
            DEFAULT_DEGREE,
            rand::thread_rng(),
        )
        .unwrap();
        Estimator::new(sampling_method)
    }
}

impl<T> From<&[T]> for Estimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    /// Performs the conversion, directly from samples.
    ///
    /// Duplicated samples will be counted to construct an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `Estimator`,
    /// but be aware that default values are given to tunable parameters.
    fn from(samples: &[T]) -> Self {
        let unnorm_distr = crate::count_dup(&samples);
        let sampling_method = Bootstrap::new(
            &unnorm_distr,
            DEFAULT_NUM_GROUPS,
            DEFAULT_DEGREE,
            rand::thread_rng(),
        )
        .unwrap();
        Estimator::new(sampling_method)
    }
}

impl<T> From<Vec<T>> for Estimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    /// Performs the conversion, directly from samples.
    ///
    /// Duplicated samples will be counted to construct an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `Estimator`,
    /// but be aware that default values are given to tunable parameters.
    fn from(samples: Vec<T>) -> Self {
        <Estimator<Bootstrap<ThreadRng>> as From<&[T]>>::from(&samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use test_case::test_case;

    use crate::test::rng;

    #[test_case([8]; "one_sample")]
    #[test_case([1, 2, 3, 4, 5, 6]; "[usize; N]")]
    #[test_case(vec!['a', 'b', 'c', 'd', 'd', 'e', 'e', 'e']; "Vec<T>")]
    #[test_case(Bootstrap::new(&[1, 2, 3, 4, 5, 6], 3, 2, rand::thread_rng()).unwrap(); "bootstrap")]
    fn from<T>(source: T)
    where
        Estimator<Bootstrap<ThreadRng>>: From<T>,
    {
        Estimator::from(source);
    }

    /// Value stability of implementation
    #[test_case([1, 2, 3, 4, 5, 6], 2.337315019221; "increasing")]
    #[test_case(vec!['a', 'b', 'c', 'd', 'd', 'e', 'e', 'e'], 2.337315019221; "letters")]
    fn entropy<T>(source: T, expected: f64)
    where
        Estimator<Bootstrap<ThreadRng>>: From<T>,
    {
        let num_groups = 3;
        let degree = 2;
        let rng = rng(1);
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();
        let mut estimator = Estimator::from(source).set_sampling_method(bootstrap);

        assert_float_eq!(estimator.entropy().unwrap(), expected, abs <= 1e-6);
    }
}

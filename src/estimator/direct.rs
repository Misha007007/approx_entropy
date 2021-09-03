use core::hash::Hash;
use polyfit_rs::polyfit_rs::polyfit;
use rand::rngs::ThreadRng;
use thiserror::Error;

use crate::{Bootstrap, SamplingMethod};

const DEFAULT_NUM_GROUPS: usize = 3;
const DEFAULT_DEGREE: usize = 2;

/// Direct entropy estimator.
///
/// Introduced by Strong et. al.[^1], it extrapolates naive entropy computations
/// of small sample size by fitting a polynomial.
///
/// This is a wrapper around a concrete sampling method,
/// extending the original paper.
///
/// # Examples
///
/// Quick estimation from an unormalized distribution.
/// ```
/// # use approx_entropy::DirectEstimator;
/// let unnorm_distr = [1, 2, 3, 4, 5, 6];
/// let mut estimator = DirectEstimator::from(unnorm_distr);
/// println!("Entropy estimation: {:?}", estimator.entropy()); // Random result
/// ```
///
/// Quick estimation from a sample.
/// ```
/// # use approx_entropy::DirectEstimator;
/// let samples = vec![1, 2, 3, 1, 1, 2, 2, 1, 3]; // samples from a random variable
/// let mut estimator = DirectEstimator::from(samples);
/// println!("Entropy estimation: {:?}", estimator.entropy()); // Random result
/// ```
///
/// [^1]: https://doi.org/10.1103/PhysRevLett.80.197
#[derive(Debug, PartialEq)]
pub struct DirectEstimator<M> {
    sampling_method: M,
}

#[derive(Error, Debug)]
#[error("Failed to estimate entropy because of numerical instability.")]
pub struct FittingError;

/// # Basic methods
impl<M> DirectEstimator<M>
where
    M: SamplingMethod,
{
    /// Constructs a new `DirectEstimator`.
    ///
    /// # Remarks
    ///
    /// The trait `From<M>` is also implemented for convenience.
    pub fn new(sampling_method: M) -> Self {
        DirectEstimator { sampling_method }
    }
    /// Estimates the entropy of the underlying distribution,
    /// known only through the empirical unnormalized distribution.
    ///
    /// # Errors
    ///
    /// If there are numerical instabilities.
    pub fn entropy(&mut self) -> Result<f64, FittingError> {
        let (inverse_size_subsamples_dup, naive_entropy_values): (Vec<_>, Vec<_>) = self
            .sampling_method
            .naive_entropies()
            .into_iter()
            .map(|(size, value)| ((1. / size as f64), value))
            .unzip();

        // Fitting a polynomial
        match polyfit(
            &inverse_size_subsamples_dup,
            &naive_entropy_values,
            self.sampling_method().degree(),
        ) {
            Ok(coefficients) => Ok(coefficients[0]),
            Err(_) => Err(FittingError),
        }
    }
}

/// # Getters
///
/// Get the underlying sampling method.
impl<M> DirectEstimator<M>
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
impl<M> DirectEstimator<M> {
    pub fn set_sampling_method<M2>(self, other: M2) -> DirectEstimator<M2>
    where
        M2: SamplingMethod,
    {
        DirectEstimator {
            sampling_method: other,
        }
    }
}

impl<M> From<M> for DirectEstimator<M>
where
    M: SamplingMethod,
{
    fn from(sampling_method: M) -> Self {
        Self::new(sampling_method)
    }
}

impl<const N: usize> From<[usize; N]> for DirectEstimator<Bootstrap<ThreadRng>> {
    /// Performs the conversion from an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `DirectEstimator`,
    /// but be aware that default values are given to tunable parameters.
    fn from(unnorm_distr: [usize; N]) -> Self {
        let sampling_method = Bootstrap::new(
            &unnorm_distr,
            DEFAULT_NUM_GROUPS,
            DEFAULT_DEGREE,
            rand::thread_rng(),
        )
        .unwrap();
        DirectEstimator::new(sampling_method)
    }
}

impl<T> From<&[T]> for DirectEstimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    /// Performs the conversion, directly from samples.
    ///
    /// Duplicated samples will be counted to construct an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `DirectEstimator`,
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
        DirectEstimator::new(sampling_method)
    }
}

impl<T> From<Vec<T>> for DirectEstimator<Bootstrap<ThreadRng>>
where
    T: Hash + Eq + Clone,
{
    /// Performs the conversion, directly from samples.
    ///
    /// Duplicated samples will be counted to construct an unnormalized distribution.
    ///
    /// # Remarks
    ///
    /// This gives an easy entry point for using `DirectEstimator`,
    /// but be aware that default values are given to tunable parameters.
    fn from(samples: Vec<T>) -> Self {
        <DirectEstimator<Bootstrap<ThreadRng>> as From<&[T]>>::from(&samples)
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
        DirectEstimator<Bootstrap<ThreadRng>>: From<T>,
    {
        DirectEstimator::from(source);
    }

    /// Value stability of implementation
    #[test_case([1, 2, 3, 4, 5, 6], 1.9511041580553; "increasing")]
    #[test_case(vec!['a', 'b', 'c', 'd', 'd', 'e', 'e', 'e'], 1.9511041580553; "letters")]
    fn entropy<T>(source: T, expected: f64)
    where
        DirectEstimator<Bootstrap<ThreadRng>>: From<T>,
    {
        let num_groups = 3;
        let degree = 2;
        let rng = rng(1);
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();
        let mut estimator = DirectEstimator::from(source).set_sampling_method(bootstrap);

        assert_float_eq!(estimator.entropy().unwrap(), expected, abs <= 1e-6);
    }
}

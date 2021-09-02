use core::hash::Hash;
use rand::rngs::ThreadRng;

use crate::{Bootstrap, SamplingMethod};

mod naive;

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
/// println!("Entropy estimation: {}", estimator.entropy()); // Random result
/// ```
///
/// Quick estimation from a sample.
/// ```
/// # use approx_entropy::Estimator;
/// let samples = vec![1, 2, 3, 1, 1, 2, 2, 1, 3]; // samples from a random variable
/// let mut estimator = Estimator::from(samples);
/// println!("Entropy estimation: {}", estimator.entropy()); // Random result
/// ```
#[derive(Debug, PartialEq)]
pub struct Estimator<M> {
    sampling_method: M,
}

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
    pub fn entropy(&mut self) -> f64 {
        let y = self.sampling_method.sample_entropy();

        let x = self.sampling_method.sample_entropy_matrix();

        // Least squares for `x ? = b`
        let x_t = x.transpose();
        let b = x_t.clone() * y;
        let a = x_t * x;

        a.lu().solve(&b).unwrap()[0] // Never fails since we know the matrix is invertible
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
    use test_case::test_case;

    #[test_case([8]; "one_sample")]
    #[test_case([1, 2, 3, 4, 5, 6]; "[usize; N]")]
    #[test_case(vec!['a', 'b', 'c', 'd', 'd', 'e', 'e', 'e']; "Vec<T>")]
    fn from<T>(source: T)
    where
        Estimator<Bootstrap<ThreadRng>>: From<T>,
    {
        Estimator::from(source);
    }
}

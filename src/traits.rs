use nalgebra::{DMatrix, DVector};
use std::error::Error;

pub trait SamplingMethod {
    type DegreeError: Error;
    type NumGroupsError: Error;
    type UnnormDistrError: Error;
    /// Returns the degree of the polynomial used to fit the naive entropy estimations.
    fn degree(&self) -> usize;
    /// Change the degree of the polynomial used to fit the naive entropy estimations.
    fn set_degree(&mut self, degree: usize) -> Result<&mut Self, Self::DegreeError>;
    /// Returns the degree of the polynomial used to fit the naive entropy estimations.
    fn num_groups(&self) -> usize;
    /// Change the number of groups.
    fn set_num_groups(&mut self, num_groups: usize) -> Result<&mut Self, Self::NumGroupsError>;
    /// Change the unnormalized distribution from which subsamples will be taken.
    fn set_unnorm_distr(
        &mut self,
        unnorm_distr: &[usize],
    ) -> Result<&mut Self, Self::UnnormDistrError>;

    /// Size of the subsamples, for each group.
    ///
    /// The first entry corresponds to the number of subsamples
    /// used to compute naive entropy in the first group.
    /// This computation will be repeated
    /// a number of times (with different subsamples) given by
    /// the first entry of the output of `samples_rep`.
    ///
    /// # Remarks
    ///
    /// The result should always be sorted from greatest to smallest.
    fn size_subsamples(&self) -> Vec<usize>;
    /// Number of repetitions the algorithm will compute naive entropy,
    /// for each possible group.
    ///
    /// The first entry corresponds to the number of time the algorithm
    /// will compute naive entropy for a subsample of size given by the first
    /// entry of the output of `size_subsamples`.
    fn samples_rep(&self) -> Vec<usize>;

    /// Size of the subsamples, with repetitions according to `samples_rep`.
    fn size_subsamples_dup(&self) -> Vec<usize> {
        let mut vec = Vec::with_capacity(self.samples_rep().into_iter().sum());
        let samples_rep = self.samples_rep();
        for (counter, size) in self.size_subsamples().into_iter().enumerate() {
            for _ in 0..samples_rep[counter] {
                vec.push(size);
            }
        }
        vec
    }

    /// Total number of naive entropy estimation used to fit a polynomial.
    ///
    /// This is equivalent to `self.samples_rep().iter().sum()`.
    ///
    /// # Remarks
    ///
    /// This might not correspond to the total number of samples of the underlying distribution.
    fn total_samples(&self) -> usize {
        self.samples_rep().iter().sum()
    }

    /// Returns all naive entropy estimations used for fitting a polynomial.
    ///
    /// This method is tightly related to `size_subsamples_dup`.
    /// Each coordinate of the output should correspond to a naive entropy estimation
    /// of a sample of size given by the same coordinate
    /// of the output vector in `size_subsamples_dup`.
    fn naive_entropies(&mut self) -> DVector<f64>;

    /// Returns a matrix used for fitting a polynomial to the values computed by `sample_entropy`.
    fn sample_entropy_matrix(&self) -> DMatrix<f64> {
        let size_subsamples_dup = self.size_subsamples_dup();
        DMatrix::<f64>::from_fn(self.total_samples(), self.degree() + 1, |r, c| {
            (size_subsamples_dup[r] as f64).powi(1 - c as i32)
        })
    }
}

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

    /// Returns all naive entropy estimations used for fitting a polynomial,
    /// as pairs `(size, value)`, where `size` is the size of the subsample used
    /// and `value` the corresponding naive entropy value.
    fn naive_entropies(&mut self) -> Vec<(usize, f64)>;
}

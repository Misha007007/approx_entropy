use nalgebra::DVector;
use thiserror::Error;

use crate::{traits::SamplingMethod, utils::count_dup, NaiveEstimator};

#[derive(Debug)]
pub struct FixedPartition {
    samples: Vec<usize>,
    size_subsamples: Vec<usize>,
    samples_rep: Vec<usize>,
    degree: usize,
}

#[derive(Error, Debug)]
pub enum ConstructionError {
    #[error(
        "Failed construction. There are too few samples (or the number of groups is too big)."
    )]
    TooFewSamples,
    #[error("Failed construction. There are too few number of groups (or the partition indicates too many elements).")]
    LowNumGroups,
    #[error("Failed construction. There are too many repetitions (or too few subsample sizes).")]
    TooManyRepetitions,
    #[error("Failed construction. There are too many subsample sizes (or too few repetitions).")]
    TooManySubsampleSizes,
    #[error("Failed construction. There are is a repetition with value zero.")]
    NullRepetition,
    #[error("Failed construction. There are is a subsample size with value zero.")]
    NullSubsampleSize,
}

///
///
/// # Remarks
///
/// Although the name is `FixedPartition`, strictly speaking it is a sub-partition:
/// there can be more samples than necessary. The extra samples are not used.
impl FixedPartition {
    /// Construct a new `FixedPartition`.
    ///
    /// # Input
    ///
    /// - `samples` corresponds to a collection of samples where
    /// repeated elements are counted as two realizations of the same element.
    /// - `size_subsamples`
    /// - `samples_rep`
    ///
    /// # Errors
    ///
    /// Reasons are given in [ConstructionError][ConstructionError]
    ///
    /// # Examples
    ///
    /// Giving samples than strictly necessary is fine.
    /// ```
    /// # use approx_entropy::FixedPartition;
    /// let samples = [1; 100]; // some random samples
    /// let size_subsamples = [3, 2, 1];
    /// let samples_rep = [1, 2, 3];
    /// let degree = 2;
    /// FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();
    /// ```
    pub fn new(
        samples: &[usize],
        size_subsamples: &[usize],
        samples_rep: &[usize],
        degree: usize,
    ) -> Result<Self, ConstructionError> {
        let num_groups = size_subsamples.len();
        if num_groups <= degree {
            return Err(ConstructionError::LowNumGroups);
        }
        if samples_rep.len() > size_subsamples.len() {
            return Err(ConstructionError::TooManyRepetitions);
        }
        if samples_rep.len() < size_subsamples.len() {
            return Err(ConstructionError::TooManySubsampleSizes);
        }
        if samples_rep.iter().any(|&rep| rep == 0) {
            return Err(ConstructionError::NullRepetition);
        }
        if size_subsamples.iter().any(|&size| size == 0) {
            return Err(ConstructionError::NullSubsampleSize);
        }
        let desired_samples = size_subsamples
            .iter()
            .zip(samples_rep)
            .map(|(size, rep)| size * rep)
            .sum();
        if samples.len() < desired_samples {
            return Err(ConstructionError::TooFewSamples);
        }
        Ok(Self::new_unchecked(
            samples,
            size_subsamples,
            samples_rep,
            degree,
        ))
    }

    /// Construct a new `Bootstrap`.
    pub fn new_unchecked(
        samples: &[usize],
        size_subsamples: &[usize],
        samples_rep: &[usize],
        degree: usize,
    ) -> Self {
        Self {
            samples: samples.to_vec(),
            size_subsamples: size_subsamples.to_vec(),
            samples_rep: samples_rep.to_vec(),
            degree,
        }
    }
}

#[derive(Error, Debug)]
#[error("Invalid number of groups: the degree is too high.")]
pub struct TooHighDegree;

#[derive(Error, Debug)]
#[error("Invalid unnormalized distribution: the total number of samples is too low.")]
pub struct Unmutable;

impl SamplingMethod for FixedPartition {
    type DegreeError = TooHighDegree;
    type NumGroupsError = Unmutable;
    type UnnormDistrError = Unmutable;

    fn degree(&self) -> usize {
        self.degree
    }

    fn set_degree(&mut self, degree: usize) -> Result<&mut Self, Self::DegreeError> {
        if self.num_groups() > degree {
            self.degree = degree;
            Ok(self)
        } else {
            Err(TooHighDegree)
        }
    }

    fn num_groups(&self) -> usize {
        self.size_subsamples.len()
    }

    /// Always errors.
    fn set_num_groups(&mut self, _num_groups: usize) -> Result<&mut Self, Self::NumGroupsError> {
        Err(Unmutable)
    }

    /// Always errors.
    fn set_unnorm_distr(
        &mut self,
        _unnorm_distr: &[usize],
    ) -> Result<&mut Self, Self::UnnormDistrError> {
        Err(Unmutable)
    }

    fn size_subsamples(&self) -> Vec<usize> {
        self.size_subsamples.clone()
    }
    fn samples_rep(&self) -> Vec<usize> {
        self.samples_rep.clone()
    }

    fn sample_entropy(&mut self) -> DVector<f64> {
        let mut y = DVector::<f64>::from_element(self.total_samples(), 0.0);
        let mut sample_long = self.samples.clone();

        let mut count = 0;
        for (group_index, group_size) in self.size_subsamples().iter().enumerate() {
            let repetitions = self.samples_rep()[group_index];
            for _ in 0..repetitions {
                let sub_sample: Vec<usize> = (0..*group_size)
                    .map(|_| sample_long.pop().unwrap()) // Naver fails by construction conditions of FixedPartition
                    .collect();
                let unnorm_distr = count_dup(&sub_sample);

                y[count] = NaiveEstimator::new(&unnorm_distr).entropy();
                count += 1;
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use nalgebra::{dmatrix, dvector};
    use test_case::test_case;

    #[test]
    fn new() {
        let samples = [0, 0, 0, 1, 1, 2];
        let size_subsamples = [3, 2, 1];
        let samples_rep = [1, 1, 1];
        let degree = 2;
        FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();
    }

    #[test]
    fn size_subsamples() {
        let samples = [0, 0, 0, 1, 1, 2];
        let size_subsamples = [3, 2, 1];
        let samples_rep = [1, 1, 1];
        let degree = 2;
        let fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

        assert_eq!(size_subsamples.to_vec(), fixed.size_subsamples());
    }

    #[test]
    fn samples_rep() {
        let samples = [0, 0, 0, 1, 1, 2];
        let size_subsamples = [3, 2, 1];
        let samples_rep = [1, 1, 1];
        let degree = 2;
        let fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

        assert_eq!(samples_rep.to_vec(), fixed.samples_rep());
    }

    #[test]
    fn total_samples() {
        let samples = [0, 0, 0, 1, 1, 2];
        let size_subsamples = [3, 2, 1];
        let samples_rep = [1, 1, 1];
        let degree = 2;
        let fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

        assert_eq!(samples_rep.len(), fixed.total_samples());
    }

    // All naive entropy estimations are zero in this case.
    #[test_case(&[0, 0, 0, 1, 1, 1, 1, 2, 2, 2], &[3, 2, 1], &[1, 2, 3], 2, dvector![0., 0., 0., 0., 0., 0.]; "all_zeros")]
    // All naive entropy estimations are ln(2) in this case.
    #[test_case(&[0, 1, 0, 1, 0, 0, 1, 1], &[4, 2], &[1, 2], 1, dvector![2.0_f64.ln(), 2.0_f64.ln(), 2.0_f64.ln()]; "all_halves")]
    fn sample_entropy(
        samples: &[usize],
        size_subsamples: &[usize],
        samples_rep: &[usize],
        degree: usize,
        expected: DVector<f64>,
    ) {
        let mut fixed = FixedPartition::new(samples, size_subsamples, samples_rep, degree).unwrap();

        for (value, expected_value) in fixed.sample_entropy().iter().zip(expected.iter()) {
            assert_float_eq!(value, expected_value, abs <= 1e-6);
        }
    }

    #[test]
    fn sample_entropy_matrix() {
        let samples = [0, 0, 0, 1, 1, 2];
        let size_subsamples = [3, 2, 1];
        let samples_rep = [1, 1, 1];
        let degree = 2;
        let fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

        let expected = dmatrix![
            3.0, 1.0, 1. / 3.;
            2.0, 1.0, 0.5;
            1.0, 1.0, 1.0;
        ];

        for (value, expected_value) in fixed.sample_entropy_matrix().iter().zip(expected.iter()) {
            assert_float_eq!(value, expected_value, abs <= 1e-6);
        }
    }
}

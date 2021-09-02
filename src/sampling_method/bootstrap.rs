use nalgebra::DVector;
use rand::{seq::SliceRandom, Rng};
use thiserror::Error;

use crate::{traits::SamplingMethod, utils::count_dup, NaiveEstimator};

#[derive(Debug)]
pub struct Bootstrap<R> {
    num_groups: usize,
    degree: usize,
    unnorm_distr: Vec<usize>,
    rng: R,
}

#[derive(Error, Debug)]
pub enum ConstructionError {
    #[error(
        "Failed construction. There are too few samples (or the number of groups is too big)."
    )]
    TooFewSamples(#[from] TooFewSamples),
    #[error("Failed construction. There are too few number of groups (or the degree is too big).")]
    LowNumGroups(#[from] LowNumGroups),
}

impl<R> Bootstrap<R>
where
    R: Rng,
{
    /// Construct a new `Bootstrap`.
    ///
    /// Notice that `unnorm_distr` must correspond to an unnormalized distribution
    /// where each entry corresponds to the number of times a specific value occured.
    /// If you need to transform a vector of samples to an unnormalized distribution,
    /// check out [`count_dup`].
    ///
    /// # Errors
    ///
    /// If the number of groups is less or equal than the degree;
    /// or if the total number of available samples is too low (for the desired number of groups).
    ///
    /// [`count_dup`]: fn.count_dup.html
    pub fn new(
        unnorm_distr: &[usize],
        num_groups: usize,
        degree: usize,
        rng: R,
    ) -> Result<Self, ConstructionError> {
        if num_groups > degree {
            let available_samples: usize = unnorm_distr.iter().sum();
            if available_samples >= 1 << num_groups {
                Ok(Bootstrap::new_unchecked(
                    unnorm_distr,
                    num_groups,
                    degree,
                    rng,
                ))
            } else {
                Err(TooFewSamples)?
            }
        } else {
            Err(LowNumGroups)?
        }
    }

    /// Construct a new `Bootstrap`.
    pub fn new_unchecked(unnorm_distr: &[usize], num_groups: usize, degree: usize, rng: R) -> Self {
        Bootstrap {
            num_groups,
            degree,
            unnorm_distr: unnorm_distr.to_vec(),
            rng,
        }
    }
}

#[derive(Error, Debug)]
#[error("Invalid degree: the number of groups is too low.")]
pub struct LowNumGroups;

#[derive(Error, Debug)]
#[error("Invalid number of groups: the degree is too high.")]
pub struct HighDegree;

#[derive(Error, Debug)]
#[error("Invalid unnormalized distribution: the total number of samples is too low.")]
pub struct TooFewSamples;

impl<R> SamplingMethod for Bootstrap<R>
where
    R: Rng,
{
    type DegreeError = HighDegree;
    type NumGroupsError = LowNumGroups;
    type UnnormDistrError = TooFewSamples;

    fn degree(&self) -> usize {
        self.degree
    }

    fn set_degree(&mut self, degree: usize) -> Result<&mut Bootstrap<R>, Self::DegreeError> {
        if self.num_groups > degree {
            self.degree = degree;
            Ok(self)
        } else {
            Err(HighDegree)
        }
    }

    fn num_groups(&self) -> usize {
        self.num_groups
    }
    fn set_num_groups(
        &mut self,
        num_groups: usize,
    ) -> Result<&mut Bootstrap<R>, Self::NumGroupsError> {
        if num_groups > self.degree {
            self.num_groups = num_groups;
            Ok(self)
        } else {
            Err(LowNumGroups)
        }
    }

    /// Change the unnormalized distribution.
    ///
    /// # Errors
    ///
    /// If there are too few samples: there must be at least `2^{num_groups}`.
    fn set_unnorm_distr(
        &mut self,
        unnorm_distr: &[usize],
    ) -> Result<&mut Self, Self::UnnormDistrError> {
        let available_samples: usize = unnorm_distr.iter().sum();
        if available_samples >= 1 << self.num_groups() {
            self.unnorm_distr = unnorm_distr.to_vec();
            Ok(self)
        } else {
            Err(TooFewSamples)
        }
    }

    fn size_subsamples(&self) -> Vec<usize> {
        let available_samples: usize = self.unnorm_distr.iter().sum();
        (0..self.num_groups())
            .map(|i| available_samples >> (i + 1))
            .collect()
    }
    fn samples_rep(&self) -> Vec<usize> {
        (0..self.num_groups()).map(|i| 1 << i).collect()
    }

    fn sample_entropy(&mut self) -> DVector<f64> {
        let mut y = DVector::<f64>::from_element(self.total_samples(), 0.0);
        let sample_long = {
            let mut vec = Vec::<usize>::new();
            for j in 0..self.unnorm_distr.len() {
                for _ in 0..self.unnorm_distr[j] {
                    vec.push(j);
                }
            }
            vec
        };

        let mut count = 0;
        let samples_rep = self.samples_rep();
        for (group_index, group_size) in self.size_subsamples().iter().enumerate() {
            for _ in 0..samples_rep[group_index] {
                let rand_sample: Vec<usize> = sample_long
                    .choose_multiple(&mut self.rng, *group_size)
                    .cloned()
                    .collect();

                let unnorm_distr = count_dup(&rand_sample);

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
    use nalgebra::dmatrix;

    #[test]
    fn new() {
        let num_groups = 3;
        let degree = 2;
        let rng = rand::thread_rng();
        Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();
    }

    #[test]
    fn size_subsamples() {
        let num_groups = 3;
        let degree = 2;
        let rng = rand::thread_rng();
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();

        assert_eq!(vec![10, 5, 2], bootstrap.size_subsamples());
    }

    #[test]
    fn samples_rep() {
        let num_groups = 3;
        let degree = 2;
        let rng = rand::thread_rng();
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();

        assert_eq!(vec![1, 2, 4], bootstrap.samples_rep());
    }

    #[test]
    fn total_samples() {
        let num_groups = 3;
        let degree = 2;
        let rng = rand::thread_rng();
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();

        assert_eq!(7, bootstrap.total_samples());
    }

    #[test]
    fn sample_entropy_matrix() {
        let num_groups = 3;
        let degree = 2;
        let rng = rand::thread_rng();
        let bootstrap = Bootstrap::new(&[1, 2, 3, 4, 5, 6], num_groups, degree, rng).unwrap();

        let expected = dmatrix![
            10.0, 1.0, 0.1;
            5.0, 1.0, 0.2;
            5.0, 1.0, 0.2;
            2.0, 1.0, 0.5;
            2.0, 1.0, 0.5;
            2.0, 1.0, 0.5;
            2.0, 1.0, 0.5
        ];

        for (value, expected_value) in bootstrap
            .sample_entropy_matrix()
            .iter()
            .zip(expected.iter())
        {
            assert_float_eq!(value, expected_value, abs <= 1e-6);
        }
    }
}

use nalgebra::DVector;
use rand::{seq::SliceRandom, Rng};
use thiserror::Error;

use crate::{traits::SamplingMethod, utils::count_dup, NaiveEstimator};

#[derive(Debug)]
pub struct Bootstrap<R> {
    sample_size: usize,
    num_groups: usize,
    degree: usize,
    unnorm_distr: Vec<usize>,
    rng: R,
}

impl<R> Bootstrap<R>
where
    R: Rng,
{
    /// Construct a new `Bootstrap`.
    ///
    /// # Errors
    ///
    /// If the number of groups is less or equal than the degree.
    pub fn new(
        unnorm_distr: &[usize],
        num_groups: usize,
        degree: usize,
        rng: R,
    ) -> Result<Self, LowNumGroups> {
        if num_groups > degree {
            Ok(Bootstrap::new_unchecked(
                unnorm_distr,
                num_groups,
                degree,
                rng,
            ))
        } else {
            Err(LowNumGroups)
        }
    }

    /// Construct a new `Bootstrap`.
    pub fn new_unchecked(unnorm_distr: &[usize], num_groups: usize, degree: usize, rng: R) -> Self {
        let sample_size = unnorm_distr.iter().sum();
        Bootstrap {
            sample_size,
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

impl<R> SamplingMethod for Bootstrap<R>
where
    R: Rng,
{
    type DegreeError = LowNumGroups;
    type NumGroupsError = HighDegree;

    fn degree(&self) -> usize {
        self.degree
    }
    fn set_num_groups(
        &mut self,
        num_groups: usize,
    ) -> Result<&mut Bootstrap<R>, Self::NumGroupsError> {
        if num_groups > self.degree {
            self.num_groups = num_groups;
            Ok(self)
        } else {
            Err(HighDegree)
        }
    }

    fn set_degree(&mut self, degree: usize) -> Result<&mut Bootstrap<R>, Self::DegreeError> {
        if self.num_groups > degree {
            self.degree = degree;
            Ok(self)
        } else {
            Err(LowNumGroups)
        }
    }

    fn set_unnorm_distr(&mut self, unnorm_distr: &[usize]) -> &mut Self {
        self.unnorm_distr = unnorm_distr.to_vec();
        self
    }

    fn sample_entropy(&mut self) -> DVector<f64> {
        println!("Sampling entropy!");
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
        for i in 0..self.size_subsamples().len() {
            let repetitions = self.samples_rep()[i];
            for _ in 0..repetitions {
                let rand_sample: Vec<usize> = sample_long
                    .choose_multiple(&mut self.rng, repetitions)
                    .cloned()
                    .collect();

                let unnorm_distr = count_dup(&rand_sample);

                y[count] = NaiveEstimator::new(&unnorm_distr).entropy();
                count += 1;
            }
        }
        y
    }
    fn size_subsamples(&self) -> Vec<usize> {
        (0..self.num_groups)
            .map(|i| self.sample_size >> (i + 1))
            .collect()
    }
    fn samples_rep(&self) -> Vec<usize> {
        (0..self.num_groups).map(|i| 1 << i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use nalgebra::dmatrix;

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

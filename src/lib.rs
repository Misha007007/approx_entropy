use crate::nalgebra::DMatrix;
use crate::nalgebra::DVector;
use crate::nalgebra::Matrix;
use nalgebra;
use nalgebra::linalg;
use noisy_float::prelude::r64;
use noisy_float::prelude::R64;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::hash::Hash;
use rand::Rng;

// integration tests are weird, do I even need them?

#[derive(Debug, PartialEq)]
pub enum SamplingMethod {
    Bootstrap{ rng: Box<dyn Rng> },
    Coherent,
}

#[derive(Debug, PartialEq)]
pub struct Estimator {
    sample_size: usize,
    num_groups: usize,
    degree: usize,
    // sample counting reps
    sample: Vec<usize>,
    pub sampling_method: SamplingMethod,
}

impl Estimator {
    pub fn new(unnorm_distr: &[usize]) -> Self {
        let all: usize = unnorm_distr.iter().sum();
        Estimator {
            sample_size: all,
            // up to adjustment
            num_groups: 3,
            // up to adjustment
            degree: 2,
            sample: unnorm_distr.to_vec(),
            sampling_method: SamplingMethod::Bootstrap { rng: rand::thread_rng() },
        }
    }

    fn size_subsamples(&self) -> Vec<usize> {
        match self.sampling_method {
            SamplingMethod::Bootstrap {_} => (0..self.num_groups)
                .map(|i| self.sample_size >> i + 1)
                .collect(),
            // up to adjustment
            SamplingMethod::Coherent => (0..self.num_groups)
                .map(|i| self.sample_size >> i + 1)
                .collect(),
        }
    }

    fn samples_rep(&self) -> Vec<usize> {
        match self.sampling_method {
            SamplingMethod::Bootstrap {_} => (0..self.num_groups).map(|i| 1 << i).collect(),
            // up to adjustment
            SamplingMethod::Coherent => (0..self.num_groups).map(|i| 1 << i).collect(),
        }
    }

    fn total_samples(&self) -> usize {
        self.samples_rep().iter().sum()
    }

    fn entropy_bootstrap<R>(&self, rng: &mut R) -> DVector<f64> 
    where
        R: Rng + ?Sized,
    {
        let mut y = DVector::<f64>::from_element(self.total_samples(), 0.0);
        let mut sample_long = Vec::<usize>::new();

        for j in 0..self.sample.len() {
            for i in 0..self.sample[j] {
                sample_long.push(j)
            }
        }

        let mut count = 0;

        for i in 0..self.size_subsamples().len() {
            for j in 0..self.samples_rep()[i] {
                let rand_sample: Vec<usize> = sample_long
                    .choose_multiple(rng, self.size_subsamples()[i])
                    .cloned()
                    .collect();

                let count_rand_sample = hash_it(&rand_sample);

                y[count] = naive_est(&count_rand_sample);
                count += 1
            }
        }

        y
    }

    fn entropy_coherent(&self) -> DVector<f64> {
        // change later
        let mut y = DVector::<f64>::from_element(self.total_samples(), 0.0);
        let mut sample_long = Vec::<usize>::new();

        for j in 0..self.sample.len() {
            for i in 0..self.sample[j] {
                sample_long.push(j)
            }
        }

        let mut rng = &mut rand::thread_rng();
        let mut count = 0;

        for i in 0..self.size_subsamples().len() {
            for j in 0..self.samples_rep()[i] {
                let rand_sample: Vec<usize> = sample_long
                    .choose_multiple(&mut rng, self.size_subsamples()[i])
                    .cloned()
                    .collect();

                let count_rand_sample = hash_it(&rand_sample);

                y[count] = naive_est(&count_rand_sample);
                count += 1
            }
        }

        y
    }

    pub fn entropy(&self) -> f64 {
        let mut y = DVector::<f64>::from_element(self.total_samples(), 0.0);

        match self.sampling_method {
            SamplingMethod::Bootstrap{ rng } => {
                y = self.entropy_bootstrap(&mut rng);
            }
            SamplingMethod::Coherent => {
                y = self.entropy_coherent();
            }
        };

        // let mut x = DVector::<f64>::from_element(self.total_samples(), 0.0);
        let mut x = Vec::new();

        for j in 0..self.degree + 1 {
            for i in 0..self.size_subsamples().len() {
                // badly implemented power
                x.push((self.size_subsamples()[i] as f64).powi((1 - j) as i32));
            }
        }

        let mut x_matrix = DMatrix::<f64>::from_vec(self.total_samples(), self.degree + 1, x);
        let mut x_t_matrix = DMatrix::transpose(&x_matrix.clone());
        let mut b: DVector<f64> = x_t_matrix.clone() * y.clone();
        let mut A: DMatrix<f64> = x_t_matrix.clone() * x_matrix.clone();
        let decompose = A.lu();
        let b_lse = decompose.solve(&b);
        b_lse.unwrap()[0]
    }
}

impl<T> From<&[T]> for Estimator
where
    T: Hash,
    T: Eq,
    T: Clone,
{
    fn from(samples: &[T]) -> Self {
        let vec = hash_it(&samples);

        Estimator::new(&vec)
    }
}

impl<T> From<Vec<T>> for Estimator
where
    T: Hash,
    T: Eq,
    T: Clone,
{
    fn from(samples: Vec<T>) -> Self {
        let vec = hash_it(&samples);

        Estimator::new(&vec)
    }
}

fn hash_it<T>(samples: &[T]) -> Vec<usize>
where
    T: Hash + Eq + Clone,
{
    let mut distribution = HashMap::<T, usize>::new();
    for i in samples {
        let count = distribution.entry(i.clone()).or_insert(0);
        *count += 1
    }

    let mut vec = Vec::<usize>::new();

    for (_, occurrence) in &distribution {
        vec.push(*occurrence)
    }

    vec
}

fn naive_est(subsample: &[usize]) -> f64 {
    let s = subsample.len();
    let mut count: f64 = 0.0;

    let all: usize = subsample.iter().sum();

    for i in 0..s {
        count -= subsample[i] as f64 * ((subsample[i] as f64) / all as f64).ln()
    }

    count / all as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use float_eq::float_eq;

    #[test]
    fn naive_est_test() {
        let sample = vec![1, 1, 1, 1];
        // ln4 ≈ 1.386294361119891
        assert_float_eq!(naive_est(&sample), 1.38629, abs <= 0.0001);

        let sample = vec![1, 1, 1, 1, 1, 1, 1, 1];
        // ln8 ≈ 2.079441541679836
        assert_float_eq!(naive_est(&sample), 2.079441541679836, abs <= 0.000001);

        let sample = vec![1, 2, 3, 4, 5, 6];
        assert_float_eq!(naive_est(&sample), 1.66237699, abs <= 0.0001);
    }

    #[test]
    fn hash_it_test() {
        let example_string = vec![
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
            "c".to_string(),
            "c".to_string(),
        ];

        assert_eq!(hash_it(&example_string), vec!(1, 2, 3));
    }

    #[test]
    fn new_test() {
        let example = Estimator {
            sample_size: 21,
            num_groups: 3,
            degree: 2,
            // sample counting reps
            sample: vec![1, 2, 3, 4, 5, 6],
            sampling_method: SamplingMethod::Bootstrap{rng: rand::thread_rng()},
        };

        assert_eq!(example, Estimator::new(&[1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn from_test() {
        let example = Estimator {
            sample_size: 6,
            num_groups: 3,
            degree: 2,
            // sample counting reps
            sample: vec![1, 2, 3],
            sampling_method: SamplingMethod::Bootstrap {_},
        };

        let example_string = vec![
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
            "c".to_string(),
            "c".to_string(),
        ];
        assert_eq!(example, Estimator::from(example_string));
    }

    #[test]
    fn size_subsamples_test() {
        let example = Estimator {
            sample_size: 21,
            num_groups: 3,
            degree: 2,
            // sample counting reps
            sample: vec![1, 2, 3, 4, 5, 6],
            sampling_method: SamplingMethod::Bootstrap {_},
        };

        assert_eq!(vec!(10, 5, 2), example.size_subsamples());
    }

    #[test]
    fn samples_rep_test() {
        let example = Estimator {
            sample_size: 21,
            num_groups: 3,
            degree: 2,
            // sample counting reps
            sample: vec![1, 2, 3, 4, 5, 6],
            sampling_method: SamplingMethod::Bootstrap {_},
        };

        assert_eq!(vec!(1, 2, 4), example.samples_rep());
    }

    #[test]
    fn total_samples_test() {
        let example = Estimator {
            sample_size: 21,
            num_groups: 3,
            degree: 2,
            // sample counting reps
            sample: vec![1, 2, 3, 4, 5, 6],
            sampling_method: SamplingMethod::Bootstrap {_},
        };
        assert_eq!(7, example.total_samples())
    }

    #[test]
    #[ignore]
    fn entropy_bootstrap_test() {
        todo!()
    }

    #[test]
    #[ignore]
    fn entropy_coherent_test() {
        todo!()
    }

    #[test]
    #[ignore]
    fn entropy_test() {
        todo!()
    }

    // #[test]
    // fn entropy_prediction_test() {
    //     let samples = vec![r64(1.0)];
    //     assert!(entropy_prediction(&samples).abs() < 1E-6);
    // }

    // #[test]
    // fn distr_test() {
    //     let samples = vec![r64(1.0), r64(2.0), r64(3.0), r64(4.0)];
    //     assert!(distr(&samples) == vec!(r64(0.25), r64(0.25), r64(0.25), r64(0.25)));
    // }
}

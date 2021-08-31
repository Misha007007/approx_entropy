//! Approximate the entropy of a distribution from few samples.

mod estimator;
mod sampling_method;
mod traits;
mod utils;

pub use estimator::{Estimator, NaiveEstimator};
pub use sampling_method::{Bootstrap, Coherent};
pub use traits::SamplingMethod;
pub use utils::count_dup;

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use float_eq::assert_float_eq;
//     use float_eq::float_eq;

//     #[test]
//     fn naive_est_test() {
//         let sample = vec![1, 1, 1, 1];
//         // ln4 ≈ 1.386294361119891
//         assert_float_eq!(naive_est(&sample), 1.38629, abs <= 0.0001);

//         let sample = vec![1, 1, 1, 1, 1, 1, 1, 1];
//         // ln8 ≈ 2.079441541679836
//         assert_float_eq!(naive_est(&sample), 2.079441541679836, abs <= 0.000001);

//         let sample = vec![1, 2, 3, 4, 5, 6];
//         assert_float_eq!(naive_est(&sample), 1.66237699, abs <= 0.0001);
//     }

//     #[test]
//     fn hash_it_test() {
//         let example_string = vec![
//             "a".to_string(),
//             "b".to_string(),
//             "b".to_string(),
//             "c".to_string(),
//             "c".to_string(),
//             "c".to_string(),
//         ];

//         assert_eq!(hash_it(&example_string), vec!(1, 2, 3));
//     }

//     #[test]
//     fn new_test() {
//         let example = Estimator {
//             sample_size: 21,
//             num_groups: 3,
//             degree: 2,
//             // sample counting reps
//             sample: vec![1, 2, 3, 4, 5, 6],
//             sampling_method: SamplingMethod::Bootstrap{rng: rand::thread_rng()},
//         };

//         assert_eq!(example, Estimator::new(&[1, 2, 3, 4, 5, 6]));
//     }

//     #[test]
//     fn from_test() {
//         let example = Estimator {
//             sample_size: 6,
//             num_groups: 3,
//             degree: 2,
//             // sample counting reps
//             sample: vec![1, 2, 3],
//             sampling_method: SamplingMethod::Bootstrap {_},
//         };

//         let example_string = vec![
//             "a".to_string(),
//             "b".to_string(),
//             "b".to_string(),
//             "c".to_string(),
//             "c".to_string(),
//             "c".to_string(),
//         ];
//         assert_eq!(example, Estimator::from(example_string));
//     }

//     #[test]
//     fn size_subsamples_test() {
//         let example = Estimator {
//             sample_size: 21,
//             num_groups: 3,
//             degree: 2,
//             // sample counting reps
//             sample: vec![1, 2, 3, 4, 5, 6],
//             sampling_method: SamplingMethod::Bootstrap {_},
//         };

//         assert_eq!(vec!(10, 5, 2), example.size_subsamples());
//     }

//     #[test]
//     fn samples_rep_test() {
//         let example = Estimator {
//             sample_size: 21,
//             num_groups: 3,
//             degree: 2,
//             // sample counting reps
//             sample: vec![1, 2, 3, 4, 5, 6],
//             sampling_method: SamplingMethod::Bootstrap {_},
//         };

//         assert_eq!(vec!(1, 2, 4), example.samples_rep());
//     }

//     #[test]
//     fn total_samples_test() {
//         let example = Estimator {
//             sample_size: 21,
//             num_groups: 3,
//             degree: 2,
//             // sample counting reps
//             sample: vec![1, 2, 3, 4, 5, 6],
//             sampling_method: SamplingMethod::Bootstrap {_},
//         };
//         assert_eq!(7, example.total_samples())
//     }

//     #[test]
//     #[ignore]
//     fn entropy_bootstrap_test() {
//         todo!()
//     }

//     #[test]
//     #[ignore]
//     fn entropy_coherent_test() {
//         todo!()
//     }

//     #[test]
//     #[ignore]
//     fn entropy_test() {
//         todo!()
//     }

//     // #[test]
//     // fn entropy_prediction_test() {
//     //     let samples = vec![r64(1.0)];
//     //     assert!(entropy_prediction(&samples).abs() < 1E-6);
//     // }

//     // #[test]
//     // fn distr_test() {
//     //     let samples = vec![r64(1.0), r64(2.0), r64(3.0), r64(4.0)];
//     //     assert!(distr(&samples) == vec!(r64(0.25), r64(0.25), r64(0.25), r64(0.25)));
//     // }
// }

//! This examples plots entropy estimation for the uniform distribution with support size `SUPPORT`.
//!
//! # Remarks
//!
//! Needs `gnuplot` installed.

use approx_entropy::{Estimator, NaiveEstimator};
use preexplorer::errors::PreexplorerError;
use preexplorer::prelude::*;
use rand::distributions::{Distribution, Uniform};

const SUPPORT: usize = 1_000;

fn main() -> Result<(), PreexplorerError> {
    let uniform = Uniform::from(0..SUPPORT);
    let size_samples: Vec<usize> = (5..12).map(|i| 1 << i).collect();
    let samples_rep: Vec<usize> = (0..size_samples.len())
        .map(|i| 1 << (size_samples.len() - i))
        .collect();

    // Simulation and estimation
    let naive_data: Vec<(f64, f64)> = sample_naive_entropy(&uniform, &size_samples, &samples_rep)
        .into_iter()
        .map(|(n, value)| ((1. / n as f64), value))
        .collect();
    let estimator_data: Vec<(f64, f64)> =
        sample_estimator_entropy(&uniform, &size_samples, &samples_rep)
            .into_iter()
            .map(|(n, value)| ((1. / n as f64), value))
            .collect();

    // Plot the data
    let (grid, simulation_values): (Vec<f64>, Vec<f64>) = naive_data.into_iter().unzip();
    let empirical = (grid, simulation_values)
        .preexplore()
        .set_title("simulation")
        .set_style("points")
        .to_owned();

    let (grid, estimation_values): (Vec<f64>, Vec<f64>) = estimator_data.into_iter().unzip();
    let estimation = (grid.clone(), estimation_values)
        .preexplore()
        .set_title("estimation")
        .set_style("points")
        .to_owned();

    let limit_values = grid.iter().map(|_| (SUPPORT as f64).ln());
    let limit = (grid.clone(), limit_values)
        .preexplore()
        .set_title("limit")
        .to_owned();

    (empirical + estimation + limit)
        .set_title("Asymptotic behaviour of the naive entropy estimator")
        .set_logx(2)
        .set_xlabel("log(1/n)")
        .set_ylabel("entropy estimation")
        .plot("uniform")?;
    Ok(())
}

/// Computes the naive entropy estimator from samples given
/// by simulating `variable`.
///
/// The output contains pairs `(n, value)`, where `n` is a
/// sample size and `value` is the corresponding naive entropy estimation.
///
/// # Panics
///
/// If `size_samples` and `samples_rep` have not the same length.
fn sample_naive_entropy<V>(
    variable: &V,
    size_samples: &Vec<usize>,
    samples_rep: &Vec<usize>,
) -> Vec<(usize, f64)>
where
    V: Distribution<usize>,
{
    assert_eq!(size_samples.len(), samples_rep.len());
    let mut vec = Vec::with_capacity(samples_rep.iter().sum());
    for (counter, sample_size) in size_samples.iter().enumerate() {
        for _ in 0..samples_rep[counter] {
            let naive_entropy_stimation =
                NaiveEstimator::new(&samples(&variable, *sample_size)).entropy();
            vec.push((*sample_size, naive_entropy_stimation));
        }
    }
    vec
}

/// Computes the naive entropy estimator from samples given
/// by simulating `variable`.
///
/// The output contains pairs `(n, value)`, where `n` is a
/// sample size and `value` is the corresponding naive entropy estimation.
///
/// # Panics
///
/// If `size_samples` and `samples_rep` have not the same length.
fn sample_estimator_entropy<V>(
    variable: &V,
    size_samples: &Vec<usize>,
    samples_rep: &Vec<usize>,
) -> Vec<(usize, f64)>
where
    V: Distribution<usize>,
{
    assert_eq!(size_samples.len(), samples_rep.len());
    let mut vec = Vec::with_capacity(samples_rep.iter().sum());
    for (counter, sample_size) in size_samples.iter().enumerate() {
        for _ in 0..samples_rep[counter] {
            let entropy_stimation = Estimator::from(samples(&variable, *sample_size)).entropy();
            vec.push((*sample_size, entropy_stimation));
        }
    }
    vec
}

/// Samples from `variable`, `number` of times.
fn samples<V: Distribution<usize>>(variable: &V, number: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..number).map(|_| variable.sample(&mut rng)).collect()
}

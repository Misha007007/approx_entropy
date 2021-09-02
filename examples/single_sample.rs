//! This examples plots what FixedPartition does with a fixed sample.
//!
//! # Remarks
//!
//! Needs `gnuplot` installed.

use approx_entropy::{FixedPartition, SamplingMethod};
use preexplorer::errors::PreexplorerError;
use preexplorer::prelude::*;
use rand::distributions::{Distribution, Uniform};

const SUPPORT: usize = 100;
const SAMPLE_SIZE: usize = 1_000_000;

fn main() -> Result<(), PreexplorerError> {
    // Construct fixed partition from sample
    let samples: Vec<usize> = Uniform::from(0..SUPPORT)
        .sample_iter(rand::thread_rng())
        .take(SAMPLE_SIZE)
        .collect();
    let size_subsamples = [
        SAMPLE_SIZE / 4,
        SAMPLE_SIZE / 8,
        SAMPLE_SIZE / 16,
        SAMPLE_SIZE / 32,
    ];
    let samples_rep = [1, 2, 4, 8];
    let degree = 2;
    let mut fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

    // Compute naive entropy estimations that will be extrapolated
    let naive_entropies = fixed.naive_entropies();

    // Plot
    todo!()

    // // Simulation and estimation
    // let naive_data: Vec<(f64, f64)> = sample_naive_entropy(&uniform, &size_samples, &samples_rep)
    //     .into_iter()
    //     .map(|(n, value)| ((1. / n as f64), value))
    //     .collect();
    // let estimator_data: Vec<(f64, f64)> =
    //     sample_estimator_entropy(&uniform, &size_samples, &samples_rep)
    //         .into_iter()
    //         .map(|(n, value)| ((1. / n as f64), value))
    //         .collect();

    // // Plot the data
    // let (grid, simulation_values): (Vec<f64>, Vec<f64>) = naive_data.into_iter().unzip();
    // let (_, estimation_values): (Vec<f64>, Vec<f64>) = estimator_data.into_iter().unzip();
    // let limit_values = grid.iter().map(|_| (SUPPORT as f64).ln());

    // let empirical = (grid.clone(), simulation_values)
    //     .preexplore()
    //     .set_title("simulation")
    //     .set_style("points")
    //     .to_owned();
    // let estimation = (grid.clone(), estimation_values)
    //     .preexplore()
    //     .set_title("estimation")
    //     .set_style("points")
    //     .to_owned();
    // let limit = (grid.clone(), limit_values)
    //     .preexplore()
    //     .set_title("limit")
    //     .to_owned();

    // (empirical + estimation + limit)
    //     .set_title("Asymptotic behaviour of the naive entropy estimator")
    //     .set_logx(2)
    //     .set_xlabel("log(1/n)")
    //     .set_ylabel("entropy estimation")
    //     .plot("uniform")?;
    // Ok(())
}

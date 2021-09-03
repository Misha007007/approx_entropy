//! This examples plots what FixedPartition does with a fixed sample.
//!
//! # Remarks
//!
//! Needs `gnuplot` installed.

use approx_entropy::{
    count_dup, DirectEstimator, Estimator, FixedPartition, NaiveEstimator, SamplingMethod,
};
use preexplorer::errors::PreexplorerError;
use preexplorer::prelude::*;
use rand::distributions::{Distribution, Uniform};

const SUPPORT: usize = 1 << 12;
const SAMPLE_SIZE: usize = 1 << 6;

fn main() -> Result<(), PreexplorerError> {
    // Construct fixed partition from sample
    let samples: Vec<usize> = Uniform::from(0..SUPPORT)
        .sample_iter(rand::thread_rng())
        .take(SAMPLE_SIZE)
        .collect();
    let size_subsamples = [SAMPLE_SIZE / 2, SAMPLE_SIZE / 4, SAMPLE_SIZE / 8];
    let samples_rep = [1, 1, 2];
    let degree = 2;
    let mut fixed = FixedPartition::new(&samples, &size_subsamples, &samples_rep, degree).unwrap();

    // Compute naive entropy estimations that will be extrapolated
    let (sizes, values): (Vec<_>, Vec<_>) = fixed.naive_entropies().into_iter().unzip();
    println!(
        "Final estimation: {:?}",
        Estimator::from(fixed.clone()).entropy()
    );
    println!(
        "Final direct estimation: {:?}",
        DirectEstimator::from(fixed).entropy()
    );
    println!(
        "Naive entropy: {:?}",
        NaiveEstimator::new(&count_dup(&samples)).unwrap().entropy()
    );
    println!("Real: {}", (SUPPORT as f64).ln());
    // Plot
    (sizes.iter().map(|s| 1. / *s as f64), values)
        .preexplore()
        .set_title("Naive entropy subsamples used by FixedPartition")
        .set_xlabel("1/n")
        .set_ylabel("entropy estimation")
        .set_style("points")
        .plot("fixed_partition")?;

    Ok(())
}

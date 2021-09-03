//! Approximate the entropy of a distribution from few samples.
//!
//!
//!
//! # Examples
//!
//! Quick estimation from an unormalized distribution.
//! ```
//! # use approx_entropy::Estimator;
//! let unnorm_distr = [1, 2, 3, 4, 5, 6];
//! let mut estimator = Estimator::from(unnorm_distr);
//! println!("Entropy estimation: {:?}", estimator.entropy()); // Random result
//! ```

mod estimator;
mod sampling_method;
mod traits;
mod utils;

pub use estimator::{DirectEstimator, Estimator, NaiveEstimator};
pub use sampling_method::{Bootstrap, FixedPartition};
pub use traits::SamplingMethod;
pub use utils::count_dup;

pub mod prelude {
    pub use crate::{
        count_dup, Bootstrap, Estimator, FixedPartition, NaiveEstimator, SamplingMethod,
    };
}

#[cfg(test)]
mod test {
    use rand::RngCore;

    /// Construct a deterministic RNG with the given seed
    pub(crate) fn rng(seed: u64) -> impl RngCore {
        // For tests, we want a statistically good, fast, reproducible RNG.
        // PCG32 will do fine, and will be easy to embed if we ever need to.
        const INC: u64 = 11634580027462260723;
        rand_pcg::Pcg32::new(seed, INC)
    }
}

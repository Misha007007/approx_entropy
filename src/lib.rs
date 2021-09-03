//! Approximate the entropy of a distribution from few samples.

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

use thiserror::Error;

#[derive(Debug)]
pub struct NaiveEstimator<'a> {
    unnorm_distr: &'a [usize],
}

#[derive(Error, Debug)]
#[error("Invalid unnormalized distribution: there must be at least one sample.")]
pub struct NullDistribution;

impl<'a> NaiveEstimator<'a> {
    pub fn new(unnorm_distr: &'a [usize]) -> Result<Self, NullDistribution> {
        if unnorm_distr.iter().sum::<usize>() == 0 {
            return Err(NullDistribution);
        }
        Ok(NaiveEstimator::new_unchecked(unnorm_distr))
    }

    pub fn new_unchecked(unnorm_distr: &'a [usize]) -> Self {
        NaiveEstimator {
            unnorm_distr: unnorm_distr,
        }
    }

    pub fn entropy(&self) -> f64 {
        let mut entropy = 0.0;

        let all = self.unnorm_distr.iter().sum::<usize>() as f64;
        for repetitions in self.unnorm_distr.iter().map(|x| *x as f64) {
            entropy -= repetitions * (repetitions.ln() - all.ln());
        }
        entropy / all
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_eq::assert_float_eq;
    use test_case::test_case;

    #[test_case(&[11], 0.; "uniform_one")]
    #[test_case(&[1; 4], 4.0_f64.ln(); "uniform_four")]
    #[test_case(&[1; 8], 8.0_f64.ln(); "uniform_eight")]
    #[test_case(&[1, 2, 3, 4, 5, 6], 1.66237699; "increasing")]
    fn entropy(unnorm_distr: &[usize], expected: f64) {
        let naive_estimator = NaiveEstimator::new(unnorm_distr).unwrap();
        assert_float_eq!(naive_estimator.entropy(), expected, abs <= 1e-6);
    }
}

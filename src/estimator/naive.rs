#[derive(Debug)]
pub struct NaiveEstimator<'a> {
    unnorm_distr: &'a [usize],
}

impl<'a> NaiveEstimator<'a> {
    // Missing test that there is some poasitive usize
    pub fn new(unnorm_distr: &'a [usize]) -> Self {
        NaiveEstimator {
            unnorm_distr: unnorm_distr,
        }
    }

    pub fn entropy(&self) -> f64 {
        let support = self.unnorm_distr.len();
        let mut count: f64 = 0.0;

        let all: usize = self.unnorm_distr.iter().sum();

        for i in 0..support {
            count -= self.unnorm_distr[i] as f64 * ((self.unnorm_distr[i] as f64) / all as f64).ln()
        }

        count / all as f64
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
        let naive_estimator = NaiveEstimator::new(unnorm_distr);
        assert_float_eq!(naive_estimator.entropy(), expected, abs <= 1e-6);
    }
}

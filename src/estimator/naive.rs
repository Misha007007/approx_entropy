#[derive(Debug)]
pub struct NaiveEstimator<'a> {
    samples: &'a [usize],
}

impl<'a> NaiveEstimator<'a> {
    pub fn new(samples: &'a [usize]) -> Self {
        NaiveEstimator { samples: samples }
    }

    pub fn entropy(&self) -> f64 {
        let s = self.samples.len();
        let mut count: f64 = 0.0;

        let all: usize = self.samples.iter().sum();

        for i in 0..s {
            count -= self.samples[i] as f64 * ((self.samples[i] as f64) / all as f64).ln()
        }

        count / all as f64
    }
}

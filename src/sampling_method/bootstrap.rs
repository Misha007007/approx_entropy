use nalgebra::DVector;
use rand::{seq::SliceRandom, Rng};

use crate::traits::SamplingMethod;

#[derive(Debug)]
pub struct Bootstrap<R> {
    rng: R,
}

impl<R> Bootstrap<R>
where
    R: Rng,
{
    pub fn new(rng: R) -> Self {
        Bootstrap { rng }
    }
}

impl<R> SamplingMethod for Bootstrap<R>
where
    R: Rng,
{
    fn sample_entropy(&mut self) -> DVector<f64> {
        todo!()
    }
    fn size_subsamples(&self) -> Vec<usize> {
        todo!()
    }
    fn samples_rep(&self) -> Vec<usize> {
        todo!()
    }
}

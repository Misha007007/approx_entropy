use nalgebra::DVector;

use crate::traits::SamplingMethod;

#[derive(Debug)]
pub struct Coherent;

impl SamplingMethod for Coherent {
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

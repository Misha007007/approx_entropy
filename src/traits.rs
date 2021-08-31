use nalgebra::DVector;

pub trait SamplingMethod {
    fn sample_entropy(&mut self) -> DVector<f64>;
    fn size_subsamples(&self) -> Vec<usize>;
    fn samples_rep(&self) -> Vec<usize>;
    fn total_samples(&self) -> usize {
        self.samples_rep().iter().sum()
    }
}

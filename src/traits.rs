use nalgebra::{DMatrix, DVector};
use std::error::Error;

pub trait SamplingMethod {
    type DegreeError: Error;
    type NumGroupsError: Error;
    fn degree(&self) -> usize;
    fn set_degree(&mut self, degree: usize) -> Result<&mut Self, Self::DegreeError>;
    fn set_num_groups(&mut self, num_groups: usize) -> Result<&mut Self, Self::NumGroupsError>;
    fn set_unnorm_distr(&mut self, unnorm_distr: &[usize]) -> &mut Self;

    fn sample_entropy(&mut self) -> DVector<f64>;
    fn size_subsamples(&self) -> Vec<usize>;
    fn samples_rep(&self) -> Vec<usize>;
    fn total_samples(&self) -> usize {
        self.samples_rep().iter().sum()
    }
    fn sample_entropy_matrix(&self) -> DMatrix<f64> {
        let size_subsamples = self.size_subsamples();
        let samples_rep = self.samples_rep();
        let undup_x = DMatrix::from_fn(size_subsamples.len(), self.degree() + 1, |r, c| {
            size_subsamples[r] as f64 / (size_subsamples[r].pow(c as u32)) as f64
        });

        let mut vec_x = Vec::new();
        for col in 0..self.degree() + 1 {
            for undup_row in 0..size_subsamples.len() {
                for _ in 0..samples_rep[undup_row] {
                    vec_x.push(undup_x[(undup_row, col)]);
                }
            }
        }
        DMatrix::<f64>::from_vec(self.total_samples(), self.degree() + 1, vec_x)
    }
}

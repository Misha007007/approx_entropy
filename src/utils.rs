use std::{collections::HashMap, hash::Hash};

/// Returns a vector containing the number of repetitions
/// of each distinct element in `samples` (without repetition).
///
/// # Remarks
///
/// The correspondance between the original element and its repetition is lost.
/// Also, there is no guarantee on the order of the output.
pub fn count_dup<T>(samples: &[T]) -> Vec<usize>
where
    T: Hash + Eq + Clone,
{
    let mut distribution = HashMap::<T, usize>::new();
    for i in samples {
        let count = distribution.entry(i.clone()).or_insert(0);
        *count += 1
    }

    let mut vec = Vec::<usize>::new();

    for (_, occurrence) in &distribution {
        vec.push(*occurrence)
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_count_dup() {
        let samples = ["a", "b", "b", "c", "c", "c"];
        let mut output = count_dup(&samples);
        output.sort();
        assert_eq!(output, vec![1, 2, 3]);
    }
}

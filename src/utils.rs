use std::{collections::HashMap, hash::Hash};

/// Returns a vector containing the number of repetitions
/// of each distinct element in `samples`.
///
/// # Remarks
///
/// There is no guarantee on the order of the output.
/// In particular, the correspondance between the original element
/// and its number of occurrances is lost.
///
/// # Examples
///
/// From samples that contain twice all elements.
/// ```
/// # use approx_entropy::count_dup;
/// let samples = ['a', 'b', 'c', 'c', 'a', 'b'];
/// assert_eq!(count_dup(&samples), vec![2, 2, 2]);
/// ```
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

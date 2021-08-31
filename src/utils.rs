use std::{collections::HashMap, hash::Hash};

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

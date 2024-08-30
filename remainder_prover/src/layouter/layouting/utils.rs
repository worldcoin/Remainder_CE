use std::collections::HashSet;

pub fn is_subset<T: Eq + std::hash::Hash>(a: &Vec<T>, b: &Vec<T>) -> bool {
    let set_a: HashSet<_> = a.iter().collect();
    let set_b: HashSet<_> = b.iter().collect();
    set_a.is_subset(&set_b)
}

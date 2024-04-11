use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{layer::LayerId, mle::dense::DenseMle};
use remainder_shared_types::Fr;

pub fn get_dummy_random_mle(num_vars: usize) -> DenseMle<Fr, Fr> {
    let mut rng = test_rng();
    let mle_vec = (0..(1 << num_vars))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None)
}

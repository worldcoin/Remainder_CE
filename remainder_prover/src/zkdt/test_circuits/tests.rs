use crate::zkdt::binary_recomp_circuit::circuits::BinaryRecompCircuitBatched;

#[cfg(test)]
mod tests {
    use std::{time::Instant, path::Path};

    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use ark_std::{test_rng, UniformRand};
    use itertools::Itertools;
    use rand::Rng;

    use crate::{zkdt::test_circuits::circuits::BatchedFSRandomCircuit, utils::get_random_mle};
    use crate::prover::tests::test_circuit;

    #[test]
    fn test_batched_random_layer_circuit() {

        let mut rng = test_rng();

        let num_vars = 2;
        let mle = get_random_mle::<Fr>(num_vars, &mut rng);
        let other_mle = get_random_mle::<Fr>(num_vars, &mut rng);
        let circuit = BatchedFSRandomCircuit::new(
            vec![mle, other_mle],
            1
        );

        test_circuit(circuit, Some(Path::new("./random_layer_circuit_proof.json")));
    }

}
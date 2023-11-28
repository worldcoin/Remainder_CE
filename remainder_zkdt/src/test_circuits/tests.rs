#[cfg(test)]
mod tests {
    use std::{path::Path};

    use remainder_shared_types::Fr;
    use ark_std::{test_rng};
    
    

    use crate::test_circuits::circuits::BatchedFSRandomCircuit;
    use remainder::prover::helpers::test_circuit;
    use remainder::utils::get_random_mle;

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

#[cfg(test)]
mod tests {
    use std::{path::Path};
    use remainder::prover::helpers::test_circuit;

    // --- Note: This test will only compile/run if we have the FS input layer stuff ---
    // TODO!(ryancao): Unfortunately I do not have expertise in this area to implement at the moment :(
    // #[test]
    // fn test_fs_input_multiset_circuit_catboost_batched() {

    //     let (BatchedCatboostMles {
    //         input_data_mle_vec,
    //         permuted_input_data_mle_vec,
    //         multiplicities_bin_decomp_mle_input_vec, ..
    //     }, (_tree_height, _input_len)) = generate_mles_batch_catboost_single_tree::<Fr>(2, Path::new("upshot_data/"));

    //     let circuit = InputMultiSetCircuit::new(
    //         input_data_mle_vec,
    //         permuted_input_data_mle_vec,
    //         multiplicities_bin_decomp_mle_input_vec,
    //     );

    //     test_circuit(circuit, None);
    // }

}

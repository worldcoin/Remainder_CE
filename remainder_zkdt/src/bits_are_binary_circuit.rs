/// Use [remainder::digits::components::BitsAreBinary].

#[cfg(test)]
mod tests {
    use crate::input_data_to_circuit_adapter::{
        convert_zkdt_circuit_data_multi_tree_into_mles, load_upshot_data_multi_tree_batch,
        MinibatchData,
    };
    use remainder_shared_types::Fr;
    use std::path::Path;

    #[test]
    fn test_zkdt_2_tree_circuit() {
        let minibatch_data = MinibatchData {
            log_sample_minibatch_size: 10,
            sample_minibatch_number: 2,
            tree_batch_size: 2,
            tree_batch_number: 0,
        };

        let trees_batched_data = load_upshot_data_multi_tree_batch::<Fr>(
            Some(minibatch_data),
            Path::new(&"upshot_data/quantized-upshot-model.json".to_string()),
            Path::new(&"upshot_data/upshot-quantized-samples.npy".to_string()),
        );

        let tree_batched_circuit_mles =
            convert_zkdt_circuit_data_multi_tree_into_mles(trees_batched_data);
    }
}

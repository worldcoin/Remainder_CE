#[cfg(test)]
mod tests {
    use crate::prover::helpers::test_circuit;
    use crate::worldcoin::circuits::build_circuit;
    use crate::worldcoin::data::{
        load_data, medium_worldcoin_data, tiny_worldcoin_data, tiny_worldcoin_data_non_power_of_two, tiny_worldcoin_data_non_power_of_two_mask_case, tiny_worldcoin_data_responses_on_threshold_boundary, WorldcoinCircuitData
    };
    use remainder_shared_types::{FieldExt, Fr};
    use std::path::Path;
    use crate::worldcoin::{WC_BASE, WC_NUM_DIGITS};

    #[test]
    fn test_worldcoin_circuit_tiny() {
        let data = tiny_worldcoin_data::<Fr>();
        dbg!(&data);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_tiny_non_power_of_two() {
        let data = tiny_worldcoin_data_non_power_of_two::<Fr>();
        dbg!(&data);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_tiny_non_power_of_two_mask_case() {
        let data = tiny_worldcoin_data_non_power_of_two_mask_case::<Fr>();
        dbg!(&data);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_responses_on_threshold_boundary() {
        let (data_false, data_true) = tiny_worldcoin_data_responses_on_threshold_boundary::<Fr>();
        dbg!(&data_false);
        test_circuit(build_circuit(data_false), None);
        dbg!(&data_true);
        test_circuit(build_circuit(data_true), None);
    }

    #[test]
    fn test_worldcoin_circuit_medium() {
        let data = medium_worldcoin_data::<Fr>();
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_iris() {
        let data: WorldcoinCircuitData<Fr, WC_BASE, WC_NUM_DIGITS> =
            load_data(Path::new("worldcoin_witness_data").to_path_buf(), false);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_mask() {
        let data: WorldcoinCircuitData<Fr, WC_BASE, WC_NUM_DIGITS> =
            load_data(Path::new("worldcoin_witness_data").to_path_buf(), true);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }
}

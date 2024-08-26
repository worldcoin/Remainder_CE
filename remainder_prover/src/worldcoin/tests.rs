#[cfg(test)]
mod tests {
    use crate::prover::helpers::test_circuit;
    use crate::worldcoin::circuits::build_circuit;
    use crate::worldcoin::data::{
        load_data, medium_worldcoin_data, tiny_worldcoin_data, WorldcoinCircuitData
    };
    use remainder_shared_types::{FieldExt, Fr};
    use std::path::Path;

    #[test]
    fn test_worldcoin_circuit_tiny() {
        let data = tiny_worldcoin_data::<Fr>();
        dbg!(&data);
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    #[test]
    fn test_worldcoin_circuit_medium() {
        let data = medium_worldcoin_data::<Fr>();
        let circuit = build_circuit(data);
        test_circuit(circuit, None);
    }

    // #[test]
    // fn test_worldcoin_circuit_iris() {
    //     let data: WorldcoinCircuitData<Fr> =
    //         load_data(Path::new("worldcoin_witness_data/iris").to_path_buf());
    //     let circuit = build_circuit(data);
    //     test_circuit(circuit, None);
    // }

    // #[test]
    // fn test_worldcoin_circuit_mask() {
    //     let data: WorldcoinCircuitData<Fr> =
    //         load_data(Path::new("worldcoin_witness_data/mask").to_path_buf());
    //     let circuit = build_circuit(data);
    //     test_circuit(circuit, None);
    // }
}

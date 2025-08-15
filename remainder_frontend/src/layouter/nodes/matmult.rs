//! A Module for adding `Matmult` Layers to components

use remainder_shared_types::Field;

use remainder::{
    circuit_layout::CircuitLocation,
    layer::{
        layer_enum::LayerDescriptionEnum,
        matmult::{MatMultLayerDescription, MatrixDescription},
        LayerId,
    },
    mle::mle_description::MleDescription,
    utils::mle::get_total_mle_indices,
};

use crate::layouter::builder::CircuitMap;

use super::{CircuitNode, CompilableNode, NodeId};

use anyhow::Result;

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct MatMultNode {
    id: NodeId,
    matrix_a: NodeId,
    rows_cols_num_vars_a: (usize, usize),
    matrix_b: NodeId,
    rows_cols_num_vars_b: (usize, usize),
    num_vars: usize,
}

impl CircuitNode for MatMultNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.matrix_a, self.matrix_b]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl MatMultNode {
    /// Constructs a new MatMultNode and computes the data it generates
    pub fn new(
        matrix_node_a: &dyn CircuitNode,
        rows_cols_num_vars_a: (usize, usize),
        matrix_node_b: &dyn CircuitNode,
        rows_cols_num_vars_b: (usize, usize),
    ) -> Self {
        assert_eq!(rows_cols_num_vars_a.1, rows_cols_num_vars_b.0);
        let num_product_vars = rows_cols_num_vars_a.0 + rows_cols_num_vars_b.1;

        Self {
            id: NodeId::new(),
            matrix_a: matrix_node_a.id(),
            rows_cols_num_vars_a,
            matrix_b: matrix_node_b.id(),
            rows_cols_num_vars_b,
            num_vars: num_product_vars,
        }
    }
}

impl<F: Field> CompilableNode<F> for MatMultNode {
    fn generate_circuit_description(
        &self,
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>> {
        let (matrix_a_location, matrix_a_num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.matrix_a)?;

        let mle_a_indices =
            get_total_mle_indices(&matrix_a_location.prefix_bits, *matrix_a_num_vars);
        let circuit_mle_a = MleDescription::new(matrix_a_location.layer_id, &mle_a_indices);

        // Matrix A and matrix B are not padded because the data from the previous layer is only stored as the raw [MultilinearExtension].
        let matrix_a = MatrixDescription::new(
            circuit_mle_a,
            self.rows_cols_num_vars_a.0,
            self.rows_cols_num_vars_a.1,
        );
        let (matrix_b_location, matrix_b_num_vars) =
            circuit_map.get_location_num_vars_from_node_id(&self.matrix_b)?;
        let mle_b_indices =
            get_total_mle_indices(&matrix_b_location.prefix_bits, *matrix_b_num_vars);
        let circuit_mle_b = MleDescription::new(matrix_b_location.layer_id, &mle_b_indices);

        // should already been padded
        let matrix_b = MatrixDescription::new(
            circuit_mle_b,
            self.rows_cols_num_vars_b.0,
            self.rows_cols_num_vars_b.1,
        );

        let matmult_layer_id = LayerId::next_layer_id();
        let matmult_layer = MatMultLayerDescription::new(matmult_layer_id, matrix_a, matrix_b);
        circuit_map.add_node_id_and_location_num_vars(
            self.id,
            (
                CircuitLocation::new(matmult_layer_id, vec![]),
                self.get_num_vars(),
            ),
        );

        Ok(vec![LayerDescriptionEnum::MatMult(matmult_layer)])
    }
}

#[cfg(test)]
mod test {
    use remainder_shared_types::{Field, Fr};

    use crate::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
    use remainder::mle::evals::MultilinearExtension;

    use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving for the matmul test circuit.
    fn build_matmul_test_circuit_description<F: Field>(
        matrix_a_num_rows_vars: usize,
        matrix_a_num_cols_vars: usize, // This is the same as `matrix_b_num_rows_vars`
        matrix_b_num_cols_vars: usize,
    ) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // All inputs are public inputs
        let public_input_layer_node = builder.add_input_layer(LayerVisibility::Public);

        // Inputs to the circuit include the "matrix A MLE" and the "matrix B MLE"
        let matrix_a_mle_shred = builder.add_input_shred(
            "Matrix A MLE",
            matrix_a_num_rows_vars + matrix_a_num_cols_vars,
            &public_input_layer_node,
        );
        let matrix_b_mle_shred = builder.add_input_shred(
            "Matrix B MLE",
            matrix_a_num_cols_vars + matrix_b_num_cols_vars,
            &public_input_layer_node,
        );
        let expected_result_mle_shred = builder.add_input_shred(
            "Expected Result MLE",
            matrix_a_num_rows_vars + matrix_b_num_cols_vars,
            &public_input_layer_node,
        );

        // Create the circuit components
        let matmult_sector = builder.add_matmult_node(
            &matrix_a_mle_shred,
            (matrix_a_num_rows_vars, matrix_a_num_cols_vars),
            &matrix_b_mle_shred,
            (matrix_a_num_cols_vars, matrix_b_num_cols_vars),
        );

        let difference_sector = builder.add_sector(matmult_sector - expected_result_mle_shred);
        builder.set_output(&difference_sector);

        builder.build().unwrap()
    }

    #[test]
    fn test_matmult_node_in_circuit() {
        // Define data + input sizes first
        // (4, 2) * (2, 2) --> (2, 2) for real sizes; take log_2 for num vars
        let matrix_a_num_rows_vars = 2;
        let matrix_a_num_cols_vars = 1;
        let matrix_b_num_rows_vars = 1;

        let matrix_a_mle = MultilinearExtension::new(vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
        ]);

        let matrix_b_mle =
            MultilinearExtension::new(vec![Fr::from(3), Fr::from(5), Fr::from(9), Fr::from(6)]);

        let expected_matrix_mle = MultilinearExtension::new(vec![
            Fr::from(3 + 2 * 9),
            Fr::from(5 + 2 * 6),
            Fr::from(9 * 3 + 10 * 9),
            Fr::from(9 * 5 + 10 * 6),
            Fr::from(13 * 3 + 9),
            Fr::from(13 * 5 + 6),
            Fr::from(3 * 3 + 10 * 9),
            Fr::from(3 * 5 + 10 * 6),
        ]);

        // Create circuit description + input helper function
        let mut circuit = build_matmul_test_circuit_description(
            matrix_a_num_rows_vars,
            matrix_a_num_cols_vars,
            matrix_b_num_rows_vars,
        );

        circuit.set_input("Matrix A MLE", matrix_a_mle);
        circuit.set_input("Matrix B MLE", matrix_b_mle);
        circuit.set_input("Expected Result MLE", expected_matrix_mle);

        let provable_circuit = circuit.finalize().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    #[test]
    fn test_non_power_of_2_matmult_node_in_circuit() {
        // Define data + input sizes first
        // (4, 2) * (2, 2) --> (2, 2) for real sizes; take log_2 for num vars
        let matrix_a_num_rows_vars = 2;
        let matrix_a_num_cols_vars = 1;
        let matrix_b_num_rows_vars = 1;

        let matrix_a_mle = MultilinearExtension::new(vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
        ]);

        let matrix_b_mle =
            MultilinearExtension::new(vec![Fr::from(3), Fr::from(5), Fr::from(9), Fr::from(6)]);

        let expected_matrix_mle = MultilinearExtension::new(vec![
            Fr::from(3 + 2 * 9),
            Fr::from(5 + 2 * 6),
            Fr::from(9 * 3 + 10 * 9),
            Fr::from(9 * 5 + 10 * 6),
            Fr::from(13 * 3 + 9),
            Fr::from(13 * 5 + 6),
        ]);

        // Create circuit description + input helper function
        let mut circuit = build_matmul_test_circuit_description(
            matrix_a_num_rows_vars,
            matrix_a_num_cols_vars,
            matrix_b_num_rows_vars,
        );

        circuit.set_input("Matrix A MLE", matrix_a_mle);
        circuit.set_input("Matrix B MLE", matrix_b_mle);
        circuit.set_input("Expected Result MLE", expected_matrix_mle);

        let provable_circuit = circuit.finalize().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}

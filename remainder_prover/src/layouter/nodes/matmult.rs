//! A Module for adding `Matmult` Layers to components

use remainder_shared_types::Field;

use crate::{
    layer::{
        layer_enum::LayerDescriptionEnum,
        matmult::{MatMultLayerDescription, MatrixDescription},
        LayerId,
    },
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
    mle::mle_description::MleDescription,
    utils::mle::get_total_mle_indices,
};

use super::{CircuitNode, CompilableNode, NodeId};

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
        matrix_node_a: &impl CircuitNode,
        rows_cols_num_vars_a: (usize, usize),
        matrix_node_b: &impl CircuitNode,
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
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError> {
        let (matrix_a_location, matrix_a_num_vars) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.matrix_a)?;

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
            circuit_description_map.get_location_num_vars_from_node_id(&self.matrix_b)?;
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
        circuit_description_map.add_node_id_and_location_num_vars(
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

    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layouter::nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, NodeId,
        },
        mle::evals::MultilinearExtension,
    };

    use super::MatMultNode;

    use remainder_shared_types::Field;
    use std::collections::HashMap;

    use crate::{
        layer::LayerId,
        prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
    };

    /// Struct which allows for easy "semantic" feeding of inputs into the
    /// test matrix multiplication circuit.
    struct MatmultTestInputs<F: Field> {
        matrix_a_mle: MultilinearExtension<F>,
        matrix_b_mle: MultilinearExtension<F>,
        expected_matrix_mle: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving for the matmul test circuit.
    fn build_matmul_test_circuit_description<F: Field>(
        matrix_a_num_rows_vars: usize,
        matrix_a_num_cols_vars: usize, // This is the same as `matrix_b_num_rows_vars`
        matrix_b_num_cols_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(MatmultTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- All inputs are public inputs ---
        let public_input_layer_node = InputLayerNode::new(None);

        // --- Inputs to the circuit include the "matrix A MLE" and the "matrix B MLE" ---
        let matrix_a_mle_shred = InputShred::new(
            matrix_a_num_rows_vars + matrix_a_num_cols_vars,
            &public_input_layer_node,
        );
        let matrix_b_mle_shred = InputShred::new(
            matrix_a_num_cols_vars + matrix_b_num_cols_vars,
            &public_input_layer_node,
        );
        let expected_result_mle_shred = InputShred::new(
            matrix_a_num_rows_vars + matrix_b_num_cols_vars,
            &public_input_layer_node,
        );

        // --- Save IDs to be used later ---
        let matrix_a_shred_id = matrix_a_mle_shred.id();
        let matrix_b_shred_id = matrix_b_mle_shred.id();
        let expected_matrix_shred_id = expected_result_mle_shred.id();

        // --- Create the circuit components ---
        let matmult_sector = MatMultNode::new(
            &matrix_a_mle_shred,
            (matrix_a_num_rows_vars, matrix_a_num_cols_vars),
            &matrix_b_mle_shred,
            (matrix_a_num_cols_vars, matrix_b_num_cols_vars),
        );

        let difference_sector = Sector::new(
            &[&matmult_sector, &expected_result_mle_shred],
            |inputs| {
                Expression::<F, AbstractExpr>::mle(inputs[0])
                    - Expression::<F, AbstractExpr>::mle(inputs[1])
            },
        );

        let output_node = OutputNode::new_zero(&difference_sector);

        // --- Generate the circuit description ---
        let all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            matrix_a_mle_shred.into(),
            matrix_b_mle_shred.into(),
            expected_result_mle_shred.into(),
            matmult_sector.into(),
            difference_sector.into(),
            output_node.into(),
        ];

        let (circuit_description, convert_input_shreds_to_input_layers) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |matmult_test_inputs: MatmultTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (matrix_a_shred_id, matmult_test_inputs.matrix_a_mle),
                (matrix_b_shred_id, matmult_test_inputs.matrix_b_mle),
                (
                    expected_matrix_shred_id,
                    matmult_test_inputs.expected_matrix_mle,
                ),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
    }

    #[test]
    fn test_matmult_node_in_circuit() {
        // --- Define data + input sizes first ---
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

        // --- Create circuit description + input helper function ---
        let (matmult_test_circuit_desc, input_helper_fn) = build_matmul_test_circuit_description(
            matrix_a_num_rows_vars,
            matrix_a_num_cols_vars,
            matrix_b_num_rows_vars,
        );

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(MatmultTestInputs {
            matrix_a_mle,
            matrix_b_mle,
            expected_matrix_mle,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(
            &matmult_test_circuit_desc,
            private_input_layers,
            &circuit_inputs,
        );
    }
}

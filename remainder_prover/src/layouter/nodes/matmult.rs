//! A Module for adding `Matmult` Layers to components

use ark_std::log2;
use remainder_shared_types::Field;

use crate::{
    expression::circuit_expr::CircuitMle,
    layer::{
        layer_enum::CircuitLayerEnum,
        matmult::{CircuitMatMultLayer, CircuitMatrix},
        LayerId,
    },
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
    utils::get_total_mle_indices,
};

use super::{CircuitNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct MatMultNod {
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
        ctx: &Context,
        matrix_node_a: &impl CircuitNode,
        rows_cols_num_vars_a: (usize, usize),
        matrix_node_b: &impl CircuitNode,
        rows_cols_num_vars_b: (usize, usize),
    ) -> Self {
        let matrix_a_mle = DenseMle::new_from_raw(
            matrix_node_a.get_data().get_evals_vector().to_vec(),
            LayerId::Layer(0),
        );
        let matrix_a = Matrix::new(matrix_a_mle, rows_cols_num_vars_a.0, rows_cols_num_vars_a.1);

        let matrix_b_mle = DenseMle::new_from_raw(
            matrix_node_b.get_data().get_evals_vector().to_vec(),
            LayerId::Layer(0),
        );
        let matrix_b = Matrix::new(matrix_b_mle, rows_cols_num_vars_b.0, rows_cols_num_vars_b.1);

        let num_product_vars = rows_cols_num_vars_a.0 + rows_cols_num_vars_b.1;

        Self {
            id: ctx.get_new_id(),
            matrix_a: matrix_node_a.id(),
            rows_cols_num_vars_a,
            matrix_b: matrix_node_b.id(),
            num_rows_cols_b,
            num_vars: num_product_vars,
        }
    }
}

impl<F: Field> CompilableNode<F> for MatMultNode {
    fn generate_circuit_description<'a>(
        &'a self,
        layer_id: &mut LayerId,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<CircuitLayerEnum<F>>, DAGError> {
        let (matrix_a_location, matrix_a_num_vars) =
            circuit_description_map.get_node(&self.matrix_a)?;

        let mle_a_indices =
            get_total_mle_indices(&matrix_a_location.prefix_bits, *matrix_a_num_vars);
        let circuit_mle_a = CircuitMle::new(matrix_a_location.layer_id.clone(), &mle_a_indices);

        // Matrix A and matrix B are not padded because the data from the previous layer is only stored as the raw [MultilinearExtension].
        let matrix_a = CircuitMatrix::new(
            mle_a,
            self.rows_cols_num_vars_a.0,
            self.rows_cols_num_vars_a.1,
        );
        let (matrix_b_location, matrix_b_data) = circuit_map.get_node(&self.matrix_b)?;

        let mle_b = DenseMle::new_with_prefix_bits(
            MultilinearExtension::new(matrix_b_data.get_evals_vector().clone()),
            matrix_b_location.layer_id,
            matrix_b_location.prefix_bits.clone(),
        );
        let (matrix_b_location, matrix_b_num_vars) =
            circuit_description_map.get_node(&self.matrix_b)?;
        let mle_b_indices =
            get_total_mle_indices(&matrix_b_location.prefix_bits, *matrix_b_num_vars);
        let circuit_mle_b = CircuitMle::new(matrix_b_location.layer_id.clone(), &mle_b_indices);

        // should already been padded
        let matrix_b = CircuitMatrix::new(
            mle_b,
            self.rows_cols_num_vars_b.0,
            self.rows_cols_num_vars_b.1,
        );

        let matmult_layer_id = layer_id.get_and_inc();
        let matmult_layer = CircuitMatMultLayer::new(matmult_layer_id, matrix_a, matrix_b);
        circuit_description_map.add_node(
            self.id,
            (
                CircuitLocation::new(matmult_layer_id, vec![]),
                self.get_num_vars(),
            ),
        );

        Ok(vec![CircuitLayerEnum::MatMult(matmult_layer)])
    }
}

#[cfg(test)]
mod test {

    use remainder_shared_types::Fr;

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layer::{
            matmult::{product_two_matrices, Matrix},
            LayerId,
        },
        layouter::{
            compiling::LayouterCircuit,
            component::ComponentSet,
            nodes::{
                circuit_inputs::{
                    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
                },
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
                CircuitNode,
            },
        },
        mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
        prover::helpers::test_circuit,
    };

    use super::MatMultNode;

    #[test]
    fn test_matmult_node_in_circuit() {
        let circuit = LayouterCircuit::new(|ctx| {
            let mle_vec_a = MultilinearExtension::new(vec![
                Fr::from(1),
                Fr::from(2),
                Fr::from(9),
                Fr::from(10),
                Fr::from(13),
                Fr::from(1),
                Fr::from(3),
                Fr::from(10),
            ]);

            let mle_vec_b =
                MultilinearExtension::new(vec![Fr::from(3), Fr::from(5), Fr::from(9), Fr::from(6)]);

            let exp_product = MultilinearExtension::new(vec![
                Fr::from(1 * 3 + 2 * 9),
                Fr::from(1 * 5 + 2 * 6),
                Fr::from(9 * 3 + 10 * 9),
                Fr::from(9 * 5 + 10 * 6),
                Fr::from(13 * 3 + 1 * 9),
                Fr::from(13 * 5 + 1 * 6),
                Fr::from(3 * 3 + 10 * 9),
                Fr::from(3 * 5 + 10 * 6),
            ]);

            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let input_matrix_a = InputShred::new(ctx, mle_vec_a.num_vars(), &input_layer);
            let input_matrix_a_data = InputShredData::new(input_matrix_a.id(), mle_vec_a);
            let input_matrix_b = InputShred::new(ctx, mle_vec_b.num_vars(), &input_layer);
            let input_matrix_b_data = InputShredData::new(input_matrix_b.id(), mle_vec_b);
            let input_matrix_product = InputShred::new(ctx, exp_product.num_vars(), &input_layer);
            let input_matrix_product_data =
                InputShredData::new(input_matrix_product.id(), exp_product);

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![
                    input_matrix_a_data,
                    input_matrix_b_data,
                    input_matrix_product_data,
                ],
                None,
            );

            let matmult_sector =
                MatMultNode::new(ctx, &input_matrix_a, (2, 1), &input_matrix_b, (1, 1));

            let difference_sector =
                Sector::new(ctx, &[&matmult_sector, &input_matrix_product], |inputs| {
                    Expression::<Fr, AbstractExpr>::mle(inputs[0])
                        - Expression::<Fr, AbstractExpr>::mle(inputs[1])
                });

            let output_node = OutputNode::new_zero(ctx, &difference_sector);

            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                    input_layer.into(),
                    input_matrix_a.into(),
                    input_matrix_b.into(),
                    input_matrix_product.into(),
                    matmult_sector.into(),
                    difference_sector.into(),
                    output_node.into(),
                ]),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None);
    }
}

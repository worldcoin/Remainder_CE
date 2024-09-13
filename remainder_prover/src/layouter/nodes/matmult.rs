//! A Module for adding `Matmult` Layers to components

use remainder_shared_types::Field;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::{
        matmult::{product_two_matrices, MatMult, Matrix},
        LayerId,
    },
    layouter::layouting::CircuitLocation,
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    prover::proof_system::ProofSystem,
};

use super::{CircuitNode, ClaimableNode, CompilableNode, Context, NodeId};

/// A Node that represents a `Gate` layer
#[derive(Clone, Debug)]
pub struct MatMultNode<F: Field> {
    id: NodeId,
    matrix_a: NodeId,
    rows_cols_num_vars_a: (usize, usize),
    matrix_b: NodeId,
    rows_cols_num_vars_b: (usize, usize),
    data: MultilinearExtension<F>,
}

impl<F: Field> CircuitNode for MatMultNode<F> {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.matrix_a, self.matrix_b]
    }
}

impl<F: Field> MatMultNode<F> {
    /// Constructs a new MatMultNode and computes the data it generates
    pub fn new(
        ctx: &Context,
        matrix_node_a: &impl ClaimableNode<F = F>,
        rows_cols_num_vars_a: (usize, usize),
        matrix_node_b: &impl ClaimableNode<F = F>,
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

        let data = MultilinearExtension::new(product_two_matrices(&matrix_a, &matrix_b));

        Self {
            id: ctx.get_new_id(),
            matrix_a: matrix_node_a.id(),
            rows_cols_num_vars_a,
            matrix_b: matrix_node_b.id(),
            rows_cols_num_vars_b,
            data,
        }
    }
}

impl<F: Field> ClaimableNode for MatMultNode<F> {
    type F = F;

    fn get_data(&self) -> &MultilinearExtension<Self::F> {
        &self.data
    }

    fn get_expr(&self) -> Expression<Self::F, AbstractExpr> {
        Expression::<F, AbstractExpr>::mle(self.id)
    }
}

impl<F: Field, Pf: ProofSystem<F, Layer = L>, L: From<MatMult<F>>> CompilableNode<F, Pf>
    for MatMultNode<F>
{
    fn compile<'a>(
        &'a self,
        witness_builder: &mut crate::layouter::compiling::WitnessBuilder<F, Pf>,
        circuit_map: &mut crate::layouter::layouting::CircuitMap<'a, F>,
    ) -> Result<(), crate::layouter::layouting::DAGError> {
        let (matrix_a_location, matrix_a_data) = circuit_map.get_node(&self.matrix_a)?;

        let mle_a = DenseMle::new_with_prefix_bits(
            MultilinearExtension::new(matrix_a_data.get_evals_vector().clone()),
            matrix_a_location.layer_id.clone(),
            matrix_a_location.prefix_bits.clone(),
        );

        // Matrix A and matrix B are not padded because the data from the previous layer is only stored as the raw [MultilinearExtension].
        let matrix_a = Matrix::new(
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

        // should already been padded
        let matrix_b = Matrix::new(
            mle_b,
            self.rows_cols_num_vars_b.0,
            self.rows_cols_num_vars_b.1,
        );

        let layer_id = witness_builder.next_layer();
        let matmult_layer = MatMult::new(layer_id, matrix_a, matrix_b);
        witness_builder.add_layer(matmult_layer.into());
        circuit_map.add_node(
            self.id,
            (CircuitLocation::new(layer_id, vec![]), &self.data),
        );

        Ok(())
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
                circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
                circuit_outputs::OutputNode,
                node_enum::NodeEnum,
                sector::Sector,
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

            let input_matrix_a = InputShred::new(ctx, mle_vec_a, &input_layer);
            let input_matrix_b = InputShred::new(ctx, mle_vec_b, &input_layer);
            let input_matrix_product = InputShred::new(ctx, exp_product, &input_layer);

            let matmult_sector =
                MatMultNode::new(ctx, &input_matrix_a, (2, 1), &input_matrix_b, (1, 1));

            let difference_sector = Sector::new(
                ctx,
                &[&matmult_sector, &input_matrix_product],
                |inputs| {
                    Expression::<Fr, AbstractExpr>::mle(inputs[0])
                        - Expression::<Fr, AbstractExpr>::mle(inputs[1])
                },
                |inputs| {
                    let data: Vec<_> = inputs[0]
                        .get_evals_vector()
                        .iter()
                        .zip(inputs[1].get_evals_vector().iter())
                        .map(|(lhs, rhs)| lhs - rhs)
                        .collect();

                    MultilinearExtension::new(data)
                },
            );

            let output_node = OutputNode::new_zero(ctx, &difference_sector);

            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_matrix_a.into(),
                input_matrix_b.into(),
                input_matrix_product.into(),
                matmult_sector.into(),
                difference_sector.into(),
                output_node.into(),
            ])
        });

        test_circuit(circuit, None);
    }
}

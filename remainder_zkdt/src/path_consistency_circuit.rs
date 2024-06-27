use itertools::Itertools;
use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

fn decision_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    (0..(size - 1)).map(|idx| (idx, idx + 1, idx)).collect_vec()
}

fn leaf_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size - 1, 0)]
}

pub struct PathCheckComponent<F: FieldExt> {
    next_node_id_sector: Sector<F>,
}

impl<F: FieldExt> PathCheckComponent<F> {
    pub fn new(
        ctx: &Context,
        decision_node_ids: impl ClaimableNode<F = F>,
        leaf_node_ids: impl ClaimableNode<F = F>,
        bin_decomp_diff_signed_bit: impl ClaimableNode<F = F>,
        num_tree_bits: usize,
        num_dataparallel_bits: usize,
    ) -> Self {
        // the next node should be 2*node_id + 1 / 2*node_id + 2,
        // depending on the value of the bin_decomp_diff_signed_bit
        // corresponding to either going left or right down the decision tree
        let next_node_id_sector = Sector::new(
            ctx,
            &[&decision_node_ids, &bin_decomp_diff_signed_bit],
            |next_node_id_inputs| {
                assert_eq!(next_node_id_inputs.len(), 2);

                let decision_node_id = next_node_id_inputs[0];
                let bin_decomp_id = next_node_id_inputs[1];

                ExprBuilder::<F>::scaled(decision_node_id.expr(), F::from(2_u64))
                    + ExprBuilder::<F>::constant(F::from(2_u64))
                    - bin_decomp_id.expr()
            },
            |data| {
                assert_eq!(data.len(), 2);
                let decision_node_ids = data[0];
                let bin_decomp_diff_signed_bit = data[1];

                let result_iter = decision_node_ids
                    .get_evals_vector()
                    .into_iter()
                    .zip(bin_decomp_diff_signed_bit.get_evals_vector().into_iter())
                    .into_iter()
                    .map(|(decision_node_id, decomp_sign_bit)| {
                        F::from(2_u64) * decision_node_id + F::from(2_u64) - decomp_sign_bit
                    })
                    .collect_vec();

                MultilinearExtension::new(result_iter)
            },
        );

        let num_var_gate = decision_node_ids.get_data().num_vars();

        let nonzero_gates_add_decision = decision_add_wiring_from_size(
            1 << (num_var_gate - num_dataparallel_bits - num_tree_bits),
        );
        let nonzero_gates_add_leaf =
            leaf_add_wiring_from_size(1 << (num_var_gate - num_dataparallel_bits - num_tree_bits));

        Self {
            next_node_id_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for PathCheckComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.next_node_id_sector.into()]
    }
}

use itertools::Itertools;
use remainder::{expression::abstract_expr::ExprBuilder, layouter::component::Component};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

// generate the wiring patterns for the gate layer (checking the decision nodes)
// note that the next_node_id_mle and node_id_mle should be one-off
fn decision_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    (0..(size - 1)).map(|idx| (idx, idx + 1, idx)).collect_vec()
}

// generate the wiring patterns for the gate layer (checking the leaf nodes)
// note that the last element of next_node_id_mle on decision nodes
// should match the first element of node_id_mle on leaf nodes
fn leaf_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size - 1, 0)]
}

/// checks that the paths that the decision node paths and the leaf node paths
/// are consistent with the binary decomposition's (signed bit): going left or right
/// down the decision tree
pub struct PathCheckComponent<F: FieldExt> {
    next_node_id_sector: Sector<F>,
}

impl<F: FieldExt> PathCheckComponent<F> {
    pub fn new(
        ctx: &Context,
        decision_node_ids: impl CircuitNode,
        leaf_node_ids: impl CircuitNode,
        bin_decomp_diff_signed_bit: impl CircuitNode,
        num_tree_bits: usize,
        num_dataparallel_bits: usize,
    ) -> Self {
        // the next node should be 2*node_id + 1 / 2*node_id + 2,
        // depending on the value of the bin_decomp_diff_signed_bit
        // corresponding to either going left or right down the decision tree
        // notice it's +1 and +2, because node_id's start with 0
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
        );

        let num_var_gate = decision_node_ids.get_num_vars();

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
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.next_node_id_sector.into()]
    }
}

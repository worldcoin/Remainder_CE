use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

/// checks that all the claimed binary decompositions are in fact binary numbers
pub struct BitsAreBinary16BitComponent<F: FieldExt> {
    bin_decomp_16_bit_is_binary_sector: Sector<F>,
}

impl<F: FieldExt> BitsAreBinary16BitComponent<F> {
    pub fn new(ctx: &Context, bin_decomp_16_bit: impl ClaimableNode<F = F>) -> Self {
        let bin_decomp_16_bit_is_binary_sector = Sector::new(
            ctx,
            &[&bin_decomp_16_bit],
            |signed_bin_decomp_mle| {
                assert_eq!(signed_bin_decomp_mle.len(), 1);

                let bin_decomp_id = signed_bin_decomp_mle[0];
                ExprBuilder::<F>::products(vec![bin_decomp_id, bin_decomp_id])
                    - bin_decomp_id.expr()
            },
            |_data| MultilinearExtension::new_zero(),
        );

        Self {
            bin_decomp_16_bit_is_binary_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for BitsAreBinary16BitComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.bin_decomp_16_bit_is_binary_sector.into()]
    }
}

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

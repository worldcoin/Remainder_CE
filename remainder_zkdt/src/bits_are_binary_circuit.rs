use remainder::{
    expression::abstract_expr::ExprBuilder,
    layouter::{component::Component, nodes::ClaimableNode},
};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

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
                    - ExprBuilder::<F>::mle(bin_decomp_id)
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

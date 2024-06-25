use remainder::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
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
    pub fn new(ctx: &Context, inputs: [&Sector<F>; 1]) -> Self {
        let inputs_as_claimable_nodes: Vec<&dyn ClaimableNode<F = F>> = inputs
            .iter()
            .map(|&sector| sector as &dyn ClaimableNode<F = F>)
            .collect();

        let bin_decomp_16_bit_is_binary_sector = Sector::new(
            ctx,
            &inputs_as_claimable_nodes,
            |signed_bin_decomp_mle| {
                assert_eq!(signed_bin_decomp_mle.len(), 1);

                let bin_decomp_id = signed_bin_decomp_mle[0];
                Expression::<F, AbstractExpr>::products(vec![bin_decomp_id, bin_decomp_id])
                    - Expression::<F, AbstractExpr>::mle(bin_decomp_id)
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

use std::iter;

use itertools::Itertools;
use remainder::{expression::abstract_expr::ExprBuilder, layouter::component::Component};
use remainder_shared_types::FieldExt;

use remainder::{
    layouter::nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode, Context},
    mle::evals::MultilinearExtension,
};

pub struct InputExpoComponent<F: FieldExt> {
    r_minus_x_powers_sectors: Vec<Sector<F>>,
    bit_exponentiation_sectors: Vec<Sector<F>>,
    product_sector: Sector<F>,
}

impl<F: FieldExt> InputExpoComponent<F> {
    pub fn new(
        ctx: &Context,
        attr_inputs: [&dyn CircuitNode; 2],
        random_inputs: [&dyn CircuitNode; 2],
        bin_decomp_inputs: [&dyn CircuitNode; 16],
    ) -> Self {
        let packing_sector_nodes = attr_inputs
            .into_iter()
            .chain(random_inputs.into_iter())
            .collect_vec();

        let input_packing_sector = Sector::new(ctx, &packing_sector_nodes, |packing_nodes| {
            // inputs [attr_id, attr_val, r, r_packing]
            // expressions = r - (x.attr_id + r_packing * x.attr_val)
            assert_eq!(packing_nodes.len(), 4);
            let attr_id = packing_nodes[0];
            let attr_val = packing_nodes[1];
            let r = packing_nodes[2];
            let r_packing = packing_nodes[3];

            r.expr() - (attr_id.expr() + ExprBuilder::<F>::products(vec![attr_val, r_packing]))
        });

        let mut r_minus_x_powers_sectors = vec![input_packing_sector];
        for _ in 0..15 {
            let last_power_sector = r_minus_x_powers_sectors.last().unwrap();

            let next_power_secotr = Sector::new(ctx, &[last_power_sector], |node| {
                // inputs [attr_id, attr_val, r, r_packing]
                // expressions = r - (x.attr_id + r_packing * x.attr_val)
                assert_eq!(node.len(), 1);

                ExprBuilder::<F>::products(vec![node[0], node[0]])
            });

            r_minus_x_powers_sectors.push(next_power_secotr);
        }

        // --- computes (r - x)^i * b_ij + (1 - b_ij) ---
        let mut bit_exponentiation_sectors = Vec::new();
        for bit in 0..15 {
            let r_minus_x_powers_sector = &r_minus_x_powers_sectors[bit];
            let bin_decomp_sector = bin_decomp_inputs[bit];

            // Takes r_minus_x_power (r-x_i)^j, outputs b_ij * (r-x_i)^j + (1-b_ij)
            let bit_exponentiation_sector = Sector::new(
                ctx,
                &[r_minus_x_powers_sector, bin_decomp_sector],
                |bit_exponentiation_inputs| {
                    assert_eq!(bit_exponentiation_inputs.len(), 2);

                    let r_minus_x_power_node = bit_exponentiation_inputs[0];
                    let bin_decomp_node = bit_exponentiation_inputs[1];

                    ExprBuilder::<F>::sum(
                        ExprBuilder::<F>::products(vec![r_minus_x_power_node, bin_decomp_node]),
                        ExprBuilder::<F>::constant(F::ONE) - bin_decomp_node.expr(),
                    )
                },
            );

            bit_exponentiation_sectors.push(bit_exponentiation_sector);
        }

        let bit_exponentiation_sectors_as_claimable_nodes = bit_exponentiation_sectors
            .iter()
            .map(|sector| sector as &dyn CircuitNode)
            .collect_vec();

        let product_sector = Sector::new(
            ctx,
            &bit_exponentiation_sectors_as_claimable_nodes,
            |product_inputs| {
                assert_eq!(product_inputs.len(), 16);

                ExprBuilder::<F>::products(product_inputs)
            },
        );

        Self {
            r_minus_x_powers_sectors,
            bit_exponentiation_sectors,
            product_sector,
        }
    }
}

impl<F: FieldExt, N> Component<N> for InputExpoComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        self.r_minus_x_powers_sectors
            .into_iter()
            .chain(self.bit_exponentiation_sectors.into_iter())
            .chain(iter::once(self.product_sector))
            .map(|sector| sector.into())
            .collect_vec()
    }
}

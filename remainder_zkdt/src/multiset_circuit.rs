use std::iter;

use itertools::Itertools;
use remainder::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{component::Component, nodes::ClaimableNode},
    mle::evals::Evaluations,
};
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
        attr_inputs: [&dyn ClaimableNode<F = F>; 2],
        random_inputs: [&dyn ClaimableNode<F = F>; 2],
        bin_decomp_inputs: [&dyn ClaimableNode<F = F>; 16],
    ) -> Self {
        let packing_sector_nodes = attr_inputs
            .into_iter()
            .chain(random_inputs.into_iter())
            .collect_vec();

        let input_packing_sector = Sector::new(
            ctx,
            &packing_sector_nodes,
            |packing_nodes| {
                // inputs [attr_id, attr_val, r, r_packing]
                // expressions = r - (x.attr_id + r_packing * x.attr_val)
                assert_eq!(packing_nodes.len(), 4);
                let attr_id = packing_nodes[0];
                let attr_val = packing_nodes[1];
                let r = packing_nodes[2];
                let r_packing = packing_nodes[3];

                Expression::<F, AbstractExpr>::mle(r)
                    - (Expression::<F, AbstractExpr>::mle(attr_id)
                        + Expression::<F, AbstractExpr>::products(vec![attr_val, r_packing]))
            },
            |data| {
                assert_eq!(data.len(), 4);
                let attr_id = data[0];
                let attr_val = data[1];
                let r = data[2];
                let r_packing = data[3];
                let num_var = data[0].num_vars();

                let result_iter = attr_id
                    .get_evals_vector()
                    .into_iter()
                    .zip(attr_val.get_evals_vector().into_iter())
                    .map(|(id, val)| {
                        r.get_evals_vector()[0] - (*id + r_packing.get_evals_vector()[0] * val)
                    })
                    .collect_vec();

                MultilinearExtension::new(Evaluations::new(num_var, result_iter))
            },
        );

        let mut r_minus_x_powers_sectors = vec![input_packing_sector];
        for _ in 0..15 {
            let last_power_sector = r_minus_x_powers_sectors.last().unwrap();

            let next_power_secotr = Sector::new(
                ctx,
                &[last_power_sector],
                |node| {
                    // inputs [attr_id, attr_val, r, r_packing]
                    // expressions = r - (x.attr_id + r_packing * x.attr_val)
                    assert_eq!(node.len(), 1);

                    Expression::<F, AbstractExpr>::products(vec![node[0], node[0]])
                },
                |data| {
                    assert_eq!(data.len(), 1);
                    let num_var = data[0].num_vars();

                    let result_iter = data[0]
                        .get_evals_vector()
                        .into_iter()
                        .map(|val| *val * val)
                        .collect_vec();

                    MultilinearExtension::new(Evaluations::new(num_var, result_iter))
                },
            );

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

                    Expression::<F, AbstractExpr>::sum(
                        Expression::<F, AbstractExpr>::products(vec![
                            r_minus_x_power_node,
                            bin_decomp_node,
                        ]),
                        Expression::<F, AbstractExpr>::constant(F::ONE)
                            - Expression::<F, AbstractExpr>::mle(bin_decomp_node),
                    )
                },
                |data| {
                    assert_eq!(data.len(), 2);
                    let num_var = data[0].num_vars();

                    let r_minus_x_power = data[0];
                    let bin_decomp = data[1];

                    let result_iter = r_minus_x_power
                        .get_evals_vector()
                        .into_iter()
                        .zip(bin_decomp.get_evals_vector().into_iter())
                        .map(|(a, b)| *a * b)
                        .collect_vec();

                    MultilinearExtension::new(Evaluations::new(num_var, result_iter))
                },
            );

            bit_exponentiation_sectors.push(bit_exponentiation_sector);
        }

        let bit_exponentiation_sectors_as_claimable_nodes = bit_exponentiation_sectors
            .iter()
            .map(|sector| sector as &dyn ClaimableNode<F = F>)
            .collect_vec();

        let product_sector = Sector::new(
            ctx,
            &bit_exponentiation_sectors_as_claimable_nodes,
            |product_inputs| {
                assert_eq!(product_inputs.len(), 16);

                Expression::<F, AbstractExpr>::products(product_inputs)
            },
            |data| {
                assert_eq!(data.len(), 16);
                let num_var = data[0].num_vars();

                let init_vec = vec![F::ZERO; data[0].get_evals_vector().len()];
                let result_iter = data.into_iter().fold(init_vec, |acc, bit_exponentiation| {
                    acc.into_iter()
                        .zip(bit_exponentiation.get_evals_vector().into_iter())
                        .map(|(a, b)| a * b)
                        .collect_vec()
                });

                MultilinearExtension::new(Evaluations::new(num_var, result_iter))
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
    N: CircuitNode + From<Sector<F>> + From<OutputNode<F>>,
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

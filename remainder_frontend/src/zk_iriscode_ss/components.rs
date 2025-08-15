use remainder_shared_types::Field;

<<<<<<< HEAD:remainder_prover/src/zk_iriscode_ss/components.rs
use crate::{
    abstract_expr::{AbstractExpression, ExprBuilder},
    layouter::builder::{CircuitBuilder, NodeRef},
};
=======
use crate::abstract_expr::{AbstractExpression, ExprBuilder};
use crate::layouter::builder::{CircuitBuilder, NodeRef};
>>>>>>> benny/extract_frontend:remainder_frontend/src/zk_iriscode_ss/components.rs

/// Components for Zk iris code computation
pub struct ZkIriscodeComponent;

impl ZkIriscodeComponent {
    /// Calculates a sum of products of two equal-length vectors of Nodes.  For example, can be used for
    /// computing a random linear combination of some nodes - in this case, the `lh_multiplicands` would
    /// be instances of [crate::layouter::nodes::fiat_shamir_challenge::FiatShamirChallengeNode].
    pub fn sum_of_products<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        lh_multiplicands: Vec<&NodeRef<F>>,
        rh_multiplicands: Vec<&NodeRef<F>>,
    ) -> NodeRef<F> {
        assert_eq!(lh_multiplicands.len(), rh_multiplicands.len());
<<<<<<< HEAD:remainder_prover/src/zk_iriscode_ss/components.rs
        let sector = builder_ref.add_sector(lh_multiplicands.iter().zip(rh_multiplicands).fold(
            AbstractExpression::<F>::constant(F::ZERO),
            |acc, (lh, rh)| acc + ExprBuilder::products(vec![lh.id(), rh.id()]),
        ));
=======
        let sector = builder_ref.add_sector(
            lh_multiplicands
                .iter()
                .zip(rh_multiplicands)
                .fold(AbstractExpression::constant(F::ZERO), |acc, (lh, rh)| {
                    acc + ExprBuilder::products(vec![lh.id(), rh.id()])
                }),
        );
>>>>>>> benny/extract_frontend:remainder_frontend/src/zk_iriscode_ss/components.rs
        sector
    }
}

#[cfg(test)]
mod test {
    use remainder_shared_types::{Field, Fr};

    use crate::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
    use remainder::{
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
    };

    use super::ZkIriscodeComponent;

    use anyhow::Result;

    fn build_sum_of_products_circuit<F: Field>() -> Result<Circuit<F>> {
        let mut builder = CircuitBuilder::<F>::new();

        let n_summands = 4;
        let rh_vector_num_vars = 1;
        let lh_vector_num_vars = 0;
        // Vectors to be summed together
        let rh_input_layer = builder.add_input_layer(LayerVisibility::Public);
        let rh_input_shreds = (0..n_summands)
            .map(|i| {
                builder.add_input_shred(
                    &format!("RH Input Shred {i}"),
                    rh_vector_num_vars,
                    &rh_input_layer,
                )
            })
            .collect::<Vec<_>>();
        // Coefficients to multiple the vectors by
        let lh_input_layer = builder.add_input_layer(LayerVisibility::Public);
        let lh_input_shreds = (0..n_summands)
            .map(|i| {
                builder.add_input_shred(
                    &format!("LH Input Shred {i}"),
                    lh_vector_num_vars,
                    &lh_input_layer,
                )
            })
            .collect::<Vec<_>>();
        let sop = ZkIriscodeComponent::sum_of_products(
            &mut builder,
            lh_input_shreds.iter().collect(),
            rh_input_shreds.iter().collect(),
        );

        let _output = builder.set_output(&sop);

        builder.build()
    }

    #[test]
    fn test_sum_of_products() {
        let mut circuit = build_sum_of_products_circuit::<Fr>().unwrap();
        [
            Fr::from(17).neg(),
            Fr::from(20).neg(),
            Fr::from(2),
            Fr::from(1),
        ]
        .into_iter()
        .enumerate()
        .for_each(|(i, elem)| {
            circuit.set_input(
                &format!("LH Input Shred {i}"),
                MultilinearExtension::new(vec![elem]),
            );
        });

        [
            vec![Fr::from(1), Fr::from(0)],
            vec![Fr::from(0), Fr::from(1)],
            vec![Fr::from(5), Fr::from(6)],
            vec![Fr::from(7), Fr::from(8)],
        ]
        .into_iter()
        .enumerate()
        .for_each(|(i, mle)| {
            circuit.set_input(
                &format!("RH Input Shred {i}"),
                MultilinearExtension::new(mle),
            );
        });

        let provable_circuit = circuit.finalize().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}

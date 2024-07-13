use remainder_shared_types::Fr;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::{
        component::ComponentSet,
        nodes::{
            circuit_inputs::{InputLayerNode, InputLayerType, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            ClaimableNode,
        },
    },
    mle::evals::{Evaluations, MultilinearExtension},
    prover::helpers::test_circuit,
};

use super::LayouterCircuit;

#[test]
fn test_basic_circuit() {
    let circuit = LayouterCircuit::new(|ctx| {
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);

        let input_shred_1 = InputShred::new(
            ctx,
            MultilinearExtension::new_from_evals(Evaluations::new(
                2,
                vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
            )),
            &input_layer,
        );

        let input_shred_2 = InputShred::new(
            ctx,
            MultilinearExtension::new_from_evals(Evaluations::new(
                2,
                vec![Fr::from(16), Fr::from(16), Fr::from(16), Fr::from(16)],
            )),
            &input_layer,
        );

        let sector_1 = Sector::new(
            ctx,
            &[&input_shred_1, &input_shred_2],
            |inputs| {
                Expression::<Fr, AbstractExpr>::mle(inputs[0])
                    + Expression::<Fr, AbstractExpr>::mle(inputs[1])
            },
            |inputs| {
                let data: Vec<_> = inputs[0]
                    .get_evals_vector()
                    .iter()
                    .zip(inputs[1].get_evals_vector().iter())
                    .map(|(lhs, rhs)| lhs + rhs)
                    .collect();

                MultilinearExtension::new_from_evals(Evaluations::new(2, data))
            },
        );

        let sector_2 = Sector::new(
            ctx,
            &[&input_shred_1, &input_shred_2],
            |inputs| Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]]),
            |inputs| {
                let data: Vec<_> = inputs[0]
                    .get_evals_vector()
                    .iter()
                    .zip(inputs[1].get_evals_vector().iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .collect();

                MultilinearExtension::new_from_evals(Evaluations::new(2, data))
            },
        );

        let out_sector = Sector::new(
            ctx,
            &[&sector_1, &sector_2],
            |inputs| Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]]),
            |inputs| {
                let data: Vec<_> = inputs[0]
                    .get_evals_vector()
                    .iter()
                    .zip(inputs[1].get_evals_vector().iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .collect();

                MultilinearExtension::new_from_evals(Evaluations::new(2, data))
            },
        );

        let output_input = InputShred::new(ctx, out_sector.get_data().clone(), &input_layer);

        let final_sector = Sector::new(
            ctx,
            &[&out_sector, &output_input],
            |inputs| {
                Expression::<Fr, AbstractExpr>::mle(inputs[0])
                    - Expression::<Fr, AbstractExpr>::mle(inputs[1])
            },
            |_| MultilinearExtension::new_sized_zero(2),
        );

        let output = OutputNode::new_zero(ctx, &final_sector);

        ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
            input_layer.into(),
            input_shred_1.into(),
            input_shred_2.into(),
            sector_1.into(),
            sector_2.into(),
            output_input.into(),
            out_sector.into(),
            final_sector.into(),
            output.into(),
        ])
    });

    test_circuit(circuit, None);
}

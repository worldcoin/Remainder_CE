use std::collections::HashMap;

use remainder_shared_types::Fr;

use crate::{
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred},
        circuit_outputs::OutputNode,
        sector::Sector,
        CircuitNode, NodeId,
    },
    mle::evals::{Evaluations, MultilinearExtension},
    prover::{generate_circuit_description, helpers::test_circuit_new},
};

#[test]
fn test_basic_circuit() {
    let input_layer = InputLayerNode::new(None);

    let input_shred_1 = InputShred::new(2, &input_layer);
    let input_shred_1_id = input_shred_1.id();
    let input_shred_1_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
    ));
    let input_shred_2 = InputShred::new(2, &input_layer);
    let input_shred_2_id = input_shred_2.id();
    let input_shred_2_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![Fr::from(16), Fr::from(16), Fr::from(16), Fr::from(16)],
    ));

    let sector_1 = Sector::new(&[&input_shred_1, &input_shred_2], |inputs| {
        Expression::<Fr, AbstractExpr>::mle(inputs[0])
            + Expression::<Fr, AbstractExpr>::mle(inputs[1])
    });

    let sector_2 = Sector::new(&[&input_shred_1, &input_shred_2], |inputs| {
        Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]])
    });

    let out_sector = Sector::new(&[&sector_1, &&sector_2], |inputs| {
        Expression::<_, AbstractExpr>::products(vec![inputs[0], inputs[1]])
    });

    let output_input = InputShred::new(out_sector.get_num_vars(), &input_layer);
    let output_input_id = output_input.id();
    let output_input_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![
            Fr::from(16 * 17),
            Fr::from(16 * 17),
            Fr::from(16 * 17),
            Fr::from(16 * 17),
        ],
    ));

    let final_sector = Sector::new(&[&&out_sector, &output_input], |inputs| {
        Expression::<Fr, AbstractExpr>::mle(inputs[0])
            - Expression::<Fr, AbstractExpr>::mle(inputs[1])
    });

    let output = OutputNode::new_zero(&final_sector);

    let all_nodes = vec![
        input_layer.into(),
        input_shred_1.into(),
        input_shred_2.into(),
        sector_1.into(),
        sector_2.into(),
        output_input.into(),
        out_sector.into(),
        final_sector.into(),
        output.into(),
    ];

    let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
        generate_circuit_description(all_nodes).unwrap();

    let input_builder = move |(input_1_data, input_2_data, output_data): (
        MultilinearExtension<Fr>,
        MultilinearExtension<Fr>,
        MultilinearExtension<Fr>,
    )| {
        let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> = HashMap::new();
        input_shred_id_to_data.insert(input_shred_1_id, input_1_data);
        input_shred_id_to_data.insert(input_shred_2_id, input_2_data);
        input_shred_id_to_data.insert(output_input_id, output_data);
        input_builder_from_shred_map(input_shred_id_to_data).unwrap()
    };

    let inputs = input_builder((input_shred_1_data, input_shred_2_data, output_input_data));
    test_circuit_new(&circ_desc, HashMap::new(), &inputs);
}

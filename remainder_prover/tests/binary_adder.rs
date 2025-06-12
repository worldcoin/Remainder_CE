use std::collections::HashMap;

use itertools::Itertools;
use remainder::{
    binary_operations::{binary_adder::BinaryAdder, logical_shift::ShiftNode},
    expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
    layer::LayerId,
    layouter::{
        component::Component,
        nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            split_node::SplitNode,
            CircuitNode, NodeId,
        },
    },
    mle::evals::MultilinearExtension,
    prover::{generate_circuit_description, prove, verify, GKRCircuitDescription},
};
use remainder_shared_types::{
    config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig, ProofConfig},
    perform_function_under_prover_config, perform_function_under_verifier_config,
    transcript::{poseidon_sponge::PoseidonSponge, Transcript, TranscriptReader, TranscriptWriter},
    Fr,
};

use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

fn build_circuit() -> (
    GKRCircuitDescription<Fr>,
    HashMap<LayerId, MultilinearExtension<Fr>>,
) {
    let input_layer = InputLayerNode::new(None);

    // LHS: [a0, a1, ..., a7]
    // RHS: [b0, b1, ..., b7]
    // Carries: [c0, d1, ..., c7]
    // Results: [r0, r1, ..., r7]
    let all_inputs = InputShred::new(5, &input_layer);
    let input_node_id = all_inputs.id();

    // Check that all input bits are binary.
    let binary_sector = Sector::<Fr>::new(&[&all_inputs], |nodes| {
        assert_eq!(nodes.len(), 1);

        let b = nodes[0];
        let b_sq = Expression::<_, AbstractExpr>::products(vec![b, b]);
        let b = b.expr();

        // b * (1 - b) = b - b^2
        b - b_sq
    });
    let binary_output = OutputNode::new_zero(&binary_sector);

    // a = [a0, ..., a7],
    // b = [b0, ..., b7],
    // c = [c0, ..., c7],
    // r = [r0, ..., r7],
    let splits = SplitNode::new(&all_inputs, 2);

    let [lhs, rhs, carries, results] = splits.try_into().unwrap();

    let mut nodes: Vec<NodeEnum<Fr>> = vec![
        input_layer.into(),
        all_inputs.into(),
        binary_sector.into(),
        binary_output.into(),
    ];

    let adder = BinaryAdder::new(&lhs, &rhs, &carries);
    let compare_sector = Sector::<Fr>::new(&[&results, &adder.get_output()], |nodes| {
        nodes[0].expr() - nodes[1].expr()
    });
    let compare_output = OutputNode::new_zero(&compare_sector);

    nodes.extend(adder.yield_nodes());
    nodes.extend([compare_sector.into(), compare_output.into()]);

    nodes.extend(vec![lhs.into(), rhs.into(), carries.into(), results.into()]);

    let (circuit_description, layer_ids_to_node_ids, circuit_description_map) =
        generate_circuit_description(nodes).unwrap();
    dbg!(&circuit_description);

    // 2. Attach input data.
    let lhs = [0, 0, 1, 0, 1, 1, 0, 1];
    let rhs = [1, 0, 0, 1, 1, 1, 1, 1];
    let carries = [0, 0, 1, 1, 1, 1, 1, 1];
    let expected_results = [1, 1, 0, 0, 1, 1, 0, 0];

    let input_mle = MultilinearExtension::new(
        lhs.into_iter()
            .chain(rhs.into_iter())
            .chain(carries.into_iter())
            .chain(expected_results.into_iter())
            .map(Fr::from)
            .collect_vec(),
    );
    // dbg!(&input_mle);

    let data_mapping: HashMap<NodeId, MultilinearExtension<Fr>> =
        vec![(input_node_id, input_mle)].into_iter().collect();

    let input_mapping = circuit_description_map
        .convert_input_shreds_to_input_layers(&layer_ids_to_node_ids, &data_mapping)
        .unwrap();
    dbg!(&input_mapping);

    (circuit_description, input_mapping)
}

fn prove_circuit(
    circuit_description: &GKRCircuitDescription<Fr>,
    input_mapping: &HashMap<LayerId, MultilinearExtension<Fr>>,
) -> (Transcript<Fr>, ProofConfig) {
    let mut transcript_writer = TranscriptWriter::<Fr, PoseidonSponge<Fr>>::new("dummy label");

    let proof_config = prove(
        input_mapping,
        &HashMap::new(),
        circuit_description,
        remainder_shared_types::circuit_hash::CircuitHashType::Poseidon,
        &mut transcript_writer,
    )
    .expect("Proving failed!");

    let proof = transcript_writer.get_transcript();
    dbg!(&proof);

    (proof, proof_config)
}

fn verify_circuit(
    circuit_description: &GKRCircuitDescription<Fr>,
    input_mapping: &HashMap<LayerId, MultilinearExtension<Fr>>,
    proof: Transcript<Fr>,
    proof_config: &ProofConfig,
) {
    let mut transcript_reader = TranscriptReader::<Fr, PoseidonSponge<Fr>>::new(proof);

    verify(
        input_mapping,
        &[],
        circuit_description,
        remainder_shared_types::circuit_hash::CircuitHashType::Poseidon,
        &mut transcript_reader,
        proof_config,
    )
    .expect("Verification Failed!");
}

#[test]
fn adder_test() {
    let _subscriber = fmt().with_max_level(Level::DEBUG).init();

    let prover_config = GKRCircuitProverConfig::memory_optimized_default();

    let (circuit_description, input_mapping) =
        perform_function_under_prover_config!(build_circuit, &prover_config,);

    let (proof, proof_config) = perform_function_under_prover_config!(
        prove_circuit,
        &prover_config,
        &circuit_description,
        &input_mapping
    );

    let verifier_config = GKRCircuitVerifierConfig::new_from_proof_config(&proof_config, true);

    perform_function_under_verifier_config!(
        verify_circuit,
        &verifier_config,
        &circuit_description,
        &input_mapping,
        proof,
        &proof_config
    );
}

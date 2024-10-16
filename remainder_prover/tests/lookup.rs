use std::{cmp::max, collections::HashMap};

use remainder::{
    input_layer::ligero_input_layer::LigeroInputLayerDescription,
    layer::LayerId,
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred},
        fiat_shamir_challenge::FiatShamirChallengeNode,
        lookup::{LookupConstraint, LookupTable},
        node_enum::NodeEnum,
        CircuitNode, Context, NodeId,
    },
    mle::evals::MultilinearExtension,
    prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
};
use remainder_ligero::ligero_structs::LigeroAuxInfo;
use remainder_shared_types::{Field, Fr};

pub mod utils;
use utils::get_total_combined_mle_num_vars;

/// Struct which allows for easy "semantic" feeding of inputs into the
/// single-shred lookup test circuit.
struct SingleShredLookupTestInputs<F: Field> {
    table_mle: MultilinearExtension<F>,
    witness_mle: MultilinearExtension<F>,
    multiplicities_mle: MultilinearExtension<F>, // Should have same size as `table_mle`
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the dataparallel selector test circuit.
///
/// Note that this function also returns the layer ID of the Ligero input layer!
fn build_single_shred_lookup_test_circuit<F: Field>(
    table_mle_num_vars: usize,
    witness_mle_num_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(SingleShredLookupTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    LayerId,
) {
    // --- Create global context manager ---
    let context = Context::new();

    // --- Lookup table is public ---
    let public_input_layer_node = InputLayerNode::new(&context, None);
    let table_mle_shred = InputShred::new(&context, table_mle_num_vars, &public_input_layer_node);

    // --- Witness values are private, as are multiplicities ---
    let ligero_input_layer_node = InputLayerNode::new(&context, None);
    let witness_mle_shred =
        InputShred::new(&context, witness_mle_num_vars, &ligero_input_layer_node);
    let multiplicities_mle_shred =
        InputShred::new(&context, table_mle_num_vars, &ligero_input_layer_node);

    // --- Save IDs to be used later ---
    let ligero_input_layer_id = ligero_input_layer_node.id();
    let table_mle_id = table_mle_shred.id();
    let witness_mle_id = witness_mle_shred.id();
    let multiplicities_mle_id = multiplicities_mle_shred.id();

    // --- Create the circuit components ---
    let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(&context, 1);
    let lookup_table =
        LookupTable::new::<F>(&context, &table_mle_shred, &fiat_shamir_challenge_node);
    let lookup_constraint = LookupConstraint::new::<F>(
        &context,
        &lookup_table,
        &witness_mle_shred,
        &multiplicities_mle_shred,
    );

    let all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        table_mle_shred.into(),
        ligero_input_layer_node.into(),
        witness_mle_shred.into(),
        multiplicities_mle_shred.into(),
        fiat_shamir_challenge_node.into(),
        lookup_table.into(),
        lookup_constraint.into(),
    ];

    let (circuit_description, convert_input_shreds_to_input_layers, input_node_id_to_layer_id_map) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: SingleShredLookupTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (table_mle_id, test_inputs.table_mle),
            (witness_mle_id, test_inputs.witness_mle),
            (multiplicities_mle_id, test_inputs.multiplicities_mle),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (
        circuit_description,
        circuit_data_fn,
        *input_node_id_to_layer_id_map
            .get(&ligero_input_layer_id)
            .unwrap(),
    )
}

/// Test the case where there is only one LookupConstraint for the LookupTable i.e. just one constrained
/// MLE.
#[test]
pub fn single_shred_test() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_NUM_VARS: usize = 2;

    // --- Input generation ---
    let table_mle = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64)]);
    let witness_mle = MultilinearExtension::new(vec![
        Fr::from(0u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
    ]);
    let multiplicities_mle = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn, ligero_input_layer_id) =
        build_single_shred_lookup_test_circuit(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(SingleShredLookupTestInputs {
        table_mle,
        witness_mle,
        multiplicities_mle,
    });

    // --- Specify private input layers (+ description and precommit) ---
    let ligero_input_layer_description_with_precommit = (
        LigeroInputLayerDescription {
            layer_id: ligero_input_layer_id,
            num_vars: max(WITNESS_MLE_NUM_VARS, TABLE_MLE_NUM_VARS) + 1,
            aux: LigeroAuxInfo::<Fr>::new(
                1 << (max(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS) + 1),
                4,
                1.0,
                None,
            ),
        },
        None,
    );
    let private_input_layers = vec![(
        ligero_input_layer_id,
        ligero_input_layer_description_with_precommit,
    )]
    .into_iter()
    .collect();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}

/// Struct which allows for easy "semantic" feeding of inputs into the
/// single-shred lookup test circuit.
struct MultiShredLookupTestInputs<F: Field> {
    table_mle: MultilinearExtension<F>,
    witness_mle_1: MultilinearExtension<F>,
    witness_mle_2: MultilinearExtension<F>,
    witness_mle_3: MultilinearExtension<F>,
    witness_mle_4: MultilinearExtension<F>,
    multiplicities_mle_1: MultilinearExtension<F>, // Should have same size as `table_mle`
    multiplicities_mle_2: MultilinearExtension<F>, // Should have same size as `table_mle`
    multiplicities_mle_3: MultilinearExtension<F>, // Should have same size as `table_mle`
    multiplicities_mle_4: MultilinearExtension<F>, // Should have same size as `table_mle`
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function allowing for ease of proving for the multi-input lookup test circuit.
///
/// Note that this function also returns the layer ID of the Ligero input layer!
fn build_multi_shred_lookup_test_circuit<F: Field>(
    table_mle_num_vars: usize,
    witness_mle_1_num_vars: usize,
    witness_mle_2_num_vars: usize,
    witness_mle_3_num_vars: usize,
    witness_mle_4_num_vars: usize,
) -> (
    GKRCircuitDescription<F>,
    impl Fn(MultiShredLookupTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    LayerId,
) {
    // --- Create global context manager ---
    let context = Context::new();

    // --- Lookup table is public ---
    let public_input_layer_node = InputLayerNode::new(&context, None);
    let table_mle_shred = InputShred::new(&context, table_mle_num_vars, &public_input_layer_node);

    // --- Witness values are private, as are multiplicities ---
    let ligero_input_layer_node = InputLayerNode::new(&context, None);

    let witness_mle_1_shred =
        InputShred::new(&context, witness_mle_1_num_vars, &ligero_input_layer_node);
    let multiplicities_mle_1_shred =
        InputShred::new(&context, table_mle_num_vars, &ligero_input_layer_node);

    let witness_mle_2_shred =
        InputShred::new(&context, witness_mle_2_num_vars, &ligero_input_layer_node);
    let multiplicities_mle_2_shred =
        InputShred::new(&context, table_mle_num_vars, &ligero_input_layer_node);

    let witness_mle_3_shred =
        InputShred::new(&context, witness_mle_3_num_vars, &ligero_input_layer_node);
    let multiplicities_mle_3_shred =
        InputShred::new(&context, table_mle_num_vars, &ligero_input_layer_node);

    let witness_mle_4_shred =
        InputShred::new(&context, witness_mle_4_num_vars, &ligero_input_layer_node);
    let multiplicities_mle_4_shred =
        InputShred::new(&context, table_mle_num_vars, &ligero_input_layer_node);

    // --- Save IDs to be used later ---
    let ligero_input_layer_id = ligero_input_layer_node.id();
    let table_mle_id = table_mle_shred.id();

    let witness_mle_1_id = witness_mle_1_shred.id();
    let multiplicities_mle_1_id = multiplicities_mle_1_shred.id();
    let witness_mle_2_id = witness_mle_2_shred.id();
    let multiplicities_mle_2_id = multiplicities_mle_2_shred.id();
    let witness_mle_3_id = witness_mle_3_shred.id();
    let multiplicities_mle_3_id = multiplicities_mle_3_shred.id();
    let witness_mle_4_id = witness_mle_4_shred.id();
    let multiplicities_mle_4_id = multiplicities_mle_4_shred.id();

    // --- Create the circuit components ---
    let fiat_shamir_challenge_node = FiatShamirChallengeNode::new(&context, 1);
    let lookup_table =
        LookupTable::new::<F>(&context, &table_mle_shred, &fiat_shamir_challenge_node);
    let lookup_constraint_1 = LookupConstraint::new::<F>(
        &context,
        &lookup_table,
        &witness_mle_1_shred,
        &multiplicities_mle_1_shred,
    );
    let lookup_constraint_2 = LookupConstraint::new::<F>(
        &context,
        &lookup_table,
        &witness_mle_2_shred,
        &multiplicities_mle_2_shred,
    );
    let lookup_constraint_3 = LookupConstraint::new::<F>(
        &context,
        &lookup_table,
        &witness_mle_3_shred,
        &multiplicities_mle_3_shred,
    );
    let lookup_constraint_4 = LookupConstraint::new::<F>(
        &context,
        &lookup_table,
        &witness_mle_4_shred,
        &multiplicities_mle_4_shred,
    );

    let all_circuit_nodes: Vec<NodeEnum<F>> = vec![
        public_input_layer_node.into(),
        table_mle_shred.into(),
        ligero_input_layer_node.into(),
        witness_mle_1_shred.into(),
        multiplicities_mle_1_shred.into(),
        witness_mle_2_shred.into(),
        multiplicities_mle_2_shred.into(),
        witness_mle_3_shred.into(),
        multiplicities_mle_3_shred.into(),
        witness_mle_4_shred.into(),
        multiplicities_mle_4_shred.into(),
        fiat_shamir_challenge_node.into(),
        lookup_table.into(),
        lookup_constraint_1.into(),
        lookup_constraint_2.into(),
        lookup_constraint_3.into(),
        lookup_constraint_4.into(),
    ];

    let (circuit_description, convert_input_shreds_to_input_layers, input_node_id_to_layer_id_map) =
        generate_circuit_description(all_circuit_nodes).unwrap();

    // --- Write closure which allows easy usage of circuit inputs ---
    let circuit_data_fn = move |test_inputs: MultiShredLookupTestInputs<F>| {
        let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
            (table_mle_id, test_inputs.table_mle),
            (witness_mle_1_id, test_inputs.witness_mle_1),
            (multiplicities_mle_1_id, test_inputs.multiplicities_mle_1),
            (witness_mle_2_id, test_inputs.witness_mle_2),
            (multiplicities_mle_2_id, test_inputs.multiplicities_mle_2),
            (witness_mle_3_id, test_inputs.witness_mle_3),
            (multiplicities_mle_3_id, test_inputs.multiplicities_mle_3),
            (witness_mle_4_id, test_inputs.witness_mle_4),
            (multiplicities_mle_4_id, test_inputs.multiplicities_mle_4),
        ]
        .into_iter()
        .collect();
        convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
    };

    (
        circuit_description,
        circuit_data_fn,
        *input_node_id_to_layer_id_map
            .get(&ligero_input_layer_id)
            .unwrap(),
    )
}

/// Test the lookup functionality when there are multiple LookupConstraints for the same LookupTable.
#[test]
pub fn multi_shred_test() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_1_NUM_VARS: usize = 2;
    const WITNESS_MLE_2_NUM_VARS: usize = 2;
    const WITNESS_MLE_3_NUM_VARS: usize = 2;
    const WITNESS_MLE_4_NUM_VARS: usize = 2;

    // --- Input generation ---
    let table_mle = MultilinearExtension::new(vec![Fr::from(3u64), Fr::from(4u64)]);
    let witness_mle_1 = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_1 = MultilinearExtension::new(vec![Fr::from(3u64), Fr::from(1u64)]);
    let witness_mle_2 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_2 = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(4u64)]);
    let witness_mle_3 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(3u64),
        Fr::from(3u64),
        Fr::from(4u64),
    ]);
    let multiplicities_mle_3 = MultilinearExtension::new(vec![Fr::from(2u64), Fr::from(2u64)]);
    let witness_mle_4 = MultilinearExtension::new(vec![
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(4u64),
        Fr::from(3u64),
    ]);
    let multiplicities_mle_4 = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn, ligero_input_layer_id) =
        build_multi_shred_lookup_test_circuit(
            TABLE_MLE_NUM_VARS,
            WITNESS_MLE_1_NUM_VARS,
            WITNESS_MLE_2_NUM_VARS,
            WITNESS_MLE_3_NUM_VARS,
            WITNESS_MLE_4_NUM_VARS,
        );

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(MultiShredLookupTestInputs {
        table_mle,
        witness_mle_1,
        witness_mle_2,
        witness_mle_3,
        witness_mle_4,
        multiplicities_mle_1,
        multiplicities_mle_2,
        multiplicities_mle_3,
        multiplicities_mle_4,
    });

    // --- Specify private input layers (+ description and precommit) ---
    // The Ligero (private) input layer in this circuit consists of eight total
    // inputs -- each of the four witnesses which are to be constrained, plus
    // the multiplicities for each of them.
    let ligero_input_layer_num_vars = get_total_combined_mle_num_vars(&vec![
        TABLE_MLE_NUM_VARS,
        TABLE_MLE_NUM_VARS,
        TABLE_MLE_NUM_VARS,
        TABLE_MLE_NUM_VARS,
        WITNESS_MLE_1_NUM_VARS,
        WITNESS_MLE_2_NUM_VARS,
        WITNESS_MLE_3_NUM_VARS,
        WITNESS_MLE_4_NUM_VARS,
    ]);
    let private_input_layers = vec![(
        ligero_input_layer_id,
        (
            LigeroInputLayerDescription {
                layer_id: ligero_input_layer_id,
                num_vars: ligero_input_layer_num_vars,
                aux: LigeroAuxInfo::new(1 << ligero_input_layer_num_vars, 4, 1.0, None),
            },
            None,
        ),
    )]
    .into_iter()
    .collect();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}

/// Test that a panic occurs when the constrained MLE contains values not in the lookup table.
#[test]
#[should_panic]
pub fn test_not_satisfied() {
    const TABLE_MLE_NUM_VARS: usize = 1;
    const WITNESS_MLE_NUM_VARS: usize = 2;

    // --- Input generation ---
    let table_mle = MultilinearExtension::new(vec![Fr::from(0u64), Fr::from(1u64)]);
    let witness_mle = MultilinearExtension::new(vec![
        Fr::from(3u64),
        Fr::from(1u64),
        Fr::from(1u64),
        Fr::from(1u64),
    ]);
    let multiplicities_mle = MultilinearExtension::new(vec![Fr::from(1u64), Fr::from(3u64)]);

    // --- Create circuit description + input helper function ---
    let (circuit_description, input_helper_fn, ligero_input_layer_id) =
        build_single_shred_lookup_test_circuit(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS);

    // --- Convert input data into circuit inputs which are assignable by prover ---
    let circuit_inputs = input_helper_fn(SingleShredLookupTestInputs {
        table_mle,
        witness_mle,
        multiplicities_mle,
    });

    // --- Specify private input layers (+ description and precommit) ---
    let ligero_input_layer_description_with_precommit = (
        LigeroInputLayerDescription {
            layer_id: ligero_input_layer_id,
            num_vars: max(WITNESS_MLE_NUM_VARS, TABLE_MLE_NUM_VARS) + 1,
            aux: LigeroAuxInfo::<Fr>::new(
                1 << (max(TABLE_MLE_NUM_VARS, WITNESS_MLE_NUM_VARS) + 1),
                4,
                1.0,
                None,
            ),
        },
        None,
    );
    let private_input_layers = vec![(
        ligero_input_layer_id,
        ligero_input_layer_description_with_precommit,
    )]
    .into_iter()
    .collect();

    // --- Prove/verify the circuit ---
    test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
}

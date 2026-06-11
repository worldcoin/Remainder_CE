use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use remainder::mle::evals::MultilinearExtension;
use hyrax::gkr::verify_hyrax_proof;
use hyrax::utils::vandermonde::VandermondeInverse;
use rand::thread_rng;
use shared_types::config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig};
use shared_types::halo2curves::ff::PrimeField;
use shared_types::pedersen::PedersenCommitter;
use shared_types::transcript::ec_transcript::ECTranscript;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::{
    Bn256Point, Fq, Fr, ff_field, perform_function_under_prover_config, perform_function_under_verifier_config
};

use poseidon::SpecRef;
use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

const NUM_LEAFS: usize = 2;
const PATH_LEN: usize = 30;

const R_F: usize = 8;
const R_P: usize = 57;
const NUM_VARS_T: usize = 2;
const T: usize = 3;
const RATE: usize = 2;

fn gen_random_leafs(num_leafs: usize) -> Vec<Fr> {
    (0..num_leafs).map(|_| Fr::random(thread_rng())).collect()
}

fn gen_random_path(path_len: usize) -> Vec<(Fr, bool)> {
    (0..path_len)
        .map(|_| (Fr::random(thread_rng()), rand::random::<bool>()))
        .collect()
}

// iv for halo2 poseidon
fn poseidon_iv() -> Fr {
    Fr::from_u128(1 << 64)
}

/// add the constants of round x
fn add_constants(builder: &mut CircuitBuilder<Fr>, state: &NodeRef<Fr>, round_consts: &Vec<NodeRef<Fr>>, round: usize) -> NodeRef<Fr> {
    builder.add_sector(state + round_consts[round].clone())
}

/// Compute x^5
fn sbox_full(builder: &mut CircuitBuilder<Fr>, base: &NodeRef<Fr>) -> NodeRef<Fr> {
    let partial_sbox_1 = builder.add_sector(base.clone() * base.clone());
    let partial_sbox_2 = builder.add_sector(partial_sbox_1.clone() * partial_sbox_1.clone());
    let sbox = builder.add_sector(base * partial_sbox_2);
    sbox
}

/// Compute x^5 only on the first element
fn sbox_part(builder: &mut CircuitBuilder<Fr>, base: &NodeRef<Fr>) -> NodeRef<Fr> {
    // Split the MLE into individual entries
    let mut elems = builder.add_split_node(&base, NUM_VARS_T);
    let first_elem = elems[0].clone();
    let partial_sbox_1 = builder.add_sector(first_elem.clone() * first_elem.clone());
    let partial_sbox_2 = builder.add_sector(partial_sbox_1.clone() * partial_sbox_1.clone());
    let first_elem_sbox = builder.add_sector(first_elem * partial_sbox_2);
    elems[0] = first_elem_sbox;
    let sbox = builder.add_sector(AbstractExpression::binary_tree_selector(elems));
    sbox
}

/// One full poseidon
fn full_poseidon(builder: &mut CircuitBuilder<Fr>, mut state: NodeRef<Fr>, round_consts: &Vec<NodeRef<Fr>>, mds_matrix: &NodeRef<Fr>) -> NodeRef<Fr> {
    // Full rounds
    for i in 0..R_F / 2 {
        // state += round_constants
        state = add_constants(builder, &state, round_consts, i);
        // state = state^5
        state = sbox_full(builder, &state);
        // state = MDS_matrix * state
        state = builder.add_matmult_node(
            mds_matrix,
            (NUM_VARS_T, NUM_VARS_T),
            &state,
            (NUM_VARS_T, 0),
        );
    }
    // Partial rounds
    for i in R_F / 2..R_F / 2 + R_P {
        // state += round_constants
        state = add_constants(builder, &state, round_consts, i);
        // state = state^5 ON THE FIRST ELEMENT ONLY
        state = sbox_part(builder, &state);
        // state = MDS_matrix * state
        state = builder.add_matmult_node(
            mds_matrix,
            (NUM_VARS_T, NUM_VARS_T),
            &state,
            (NUM_VARS_T, 0),
        );
    }
    // Full rounds
    for i in R_F / 2 + R_P..R_F + R_P {
        // state += round_constants
        state = add_constants(builder, &state, round_consts, i);
        // state = state^5
        state = sbox_full(builder, &state);
        // state = MDS_matrix * state
        state = builder.add_matmult_node(
            mds_matrix,
            (NUM_VARS_T, NUM_VARS_T),
            &state,
            (NUM_VARS_T, 0),
        );
    }
    state
}

/// The entire merkle path
/// path is of (internal_node, branch), both are encrypted
/// branch = 0 ==> current node is on the left, branch = 1 ==> current node is on the right
fn merkle_path(builder: &mut CircuitBuilder<Fr>, leafs: Vec<NodeRef<Fr>>, path: Vec<(NodeRef<Fr>, NodeRef<Fr>)>, round_consts: &Vec<NodeRef<Fr>>, mds_matrix: &NodeRef<Fr>) -> NodeRef<Fr> {
    assert!(leafs.len() <= T - 1);
    let pad_len = T.next_power_of_two();
    let iv = builder.add_sector(AbstractExpression::Constant(poseidon_iv()));
    let zero = builder.add_sector(AbstractExpression::Constant(Fr::zero()));
    // breakdown leaf and add IV
    let leaf_len = leafs.len();
    let leaf_concat = [vec![iv.clone()], leafs, vec![zero.clone(); pad_len - leaf_len - 1]].concat();
    let leaf_state = builder.add_sector(AbstractExpression::binary_tree_selector(leaf_concat));

    // hash leaf
    let leaf_state = full_poseidon(builder, leaf_state, round_consts, mds_matrix);
    // DO NOT REMOVE THIS!!! Applying split node directly after mat mult causes bug in the circuit map creator.
    let leaf_state = builder.add_sector(leaf_state + Fr::zero());
    let mut node_hash = builder.add_split_node(&leaf_state, NUM_VARS_T)[0].clone();

    for (internal_node, branch) in path {
        // branch is binary
        let binary_sector = builder.add_sector(branch.clone() * branch.clone() - branch.clone());
        builder.set_output(&binary_sector);
        // create left and right node
        let left_node = builder.add_sector(branch.clone() * internal_node.clone() + (AbstractExpression::Constant(Fr::one()) - branch.clone()) * node_hash.clone());
        let right_node = builder.add_sector(branch.clone() * node_hash.clone() + (AbstractExpression::Constant(Fr::one()) - branch) * internal_node);
        let node_concat = [vec![iv.clone(), left_node, right_node], vec![zero.clone(); pad_len - 3]].concat();
        let node_state = builder.add_sector(AbstractExpression::binary_tree_selector(node_concat));
        // hash internal node
        let node_state = full_poseidon(builder, node_state, round_consts, mds_matrix);
        let node_state = builder.add_sector(node_state + Fr::zero());
        node_hash = builder.add_split_node(&node_state, NUM_VARS_T)[0].clone();   
    }
    node_hash
}

/// path_len is the number of internal nodes we provide
fn build_circuit(num_leafs: usize, path_len: usize) -> Circuit<Fr> {
    assert!(num_leafs <= T - 1);
    let mut builder = CircuitBuilder::<Fr>::new();

    let cnst_layer = builder.add_input_layer("Constants", LayerVisibility::Public);
    let mds_layer = builder.add_input_layer("MDS", LayerVisibility::Public);
    let content_layer =
        builder.add_input_layer("Content input layer", LayerVisibility::Committed);
    let expected_hash_layer =
        builder.add_input_layer("Expected hash", LayerVisibility::Public);

    // Parameters
    let round_consts = (0..R_F + R_P)
        .map(|i| {
            let round_const_shred = builder.add_input_shred(
                &format!("Round constant {}", i),
                NUM_VARS_T,
                &cnst_layer,
            );
            round_const_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();
    let mds_matrix = builder.add_input_shred("MDS Matrix", 2 * NUM_VARS_T, &mds_layer);

    // Full poseidon
    // Initial state is content
    let leafs = (0..num_leafs)
        .map(|i| {
            let leaf_shred = builder.add_input_shred(
                &format!("Leaf {}", i),
                0,
                &content_layer,
            );
            leaf_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();
    let paths = (0..path_len)
        .map(|i| {
            let internal_node_shred = builder.add_input_shred(
                &format!("Internal node {}", i),
                0,
                &content_layer,
            );
            let branch_shred = builder.add_input_shred(
                &format!("Branch {}", i),
                0,
                &content_layer,
            );
            (internal_node_shred, branch_shred)
        })
        .collect::<Vec<(NodeRef<Fr>, NodeRef<Fr>)>>();

    // Generate root hash circuit
    let root_hash = merkle_path(&mut builder, leafs, paths, &round_consts, &mds_matrix);

    // Output hash
    let expected_root_hash =
        builder.add_input_shred("Expected root hash", 0, &expected_hash_layer);
    let subtraction_sector = builder.add_sector(root_hash - expected_root_hash);
    builder.set_output(&subtraction_sector);

    builder.build().expect("Failed to build circuit")
}

// generate poseidon tests
// if T is not a power of 2, pad inputs, constants, and mds matrix with zeros to the next power of 2
// MLE automatically adds trailing zeros, so we only need to pad MDS
// each internal node includes a branch value which is false if the current node is on the left and true if it's on the right
fn gen_merkle_test(leafs: Vec<Fr>, paths: Vec<(Fr, bool)>) -> (
    Vec<MultilinearExtension<Fr>>, // constants,
    MultilinearExtension<Fr>, // mds matrix,
    Vec<MultilinearExtension<Fr>>, // leafs
    Vec<MultilinearExtension<Fr>>, // internal nodes
    Vec<MultilinearExtension<Fr>>, // branches
    Vec<MultilinearExtension<Fr>>, // hashes
    MultilinearExtension<Fr>, // root hash
) {
    assert_eq!(leafs.len(), NUM_LEAFS);
    let pad_len = T.next_power_of_two();
    // specs
    let spec = SpecRef::<Fr, T, RATE>::new(R_F, R_P);
    let (constants, mds_matrix) = (spec.constants(), spec.mds_matrices());
    let constants = constants.into_iter().map(|round_consts| {
        MultilinearExtension::new(round_consts.to_vec())
    }).collect();
    let mds_matrix = MultilinearExtension::new(
        mds_matrix.as_vec().into_iter().map(|v| {
            // pad each row of MDS to the next power of 2
            let mut padded_v = v.to_vec();
            padded_v.resize(pad_len, Fr::zero());
            padded_v
        }).flatten().collect()
    );
    // inputs / outputs
    let mut poseidon = poseidon::Poseidon::<Fr, T, RATE>::new(R_F, R_P);
    poseidon.update(&leafs);
    let leaf_data = leafs.into_iter().map(|leaf| MultilinearExtension::new(vec![leaf])).collect();
    let mut internal_node_data = Vec::new();
    let mut branch_data = Vec::new();
    let mut hash_data = Vec::new();
    let mut next_hash = poseidon.state()[0];
    for (p, branch) in paths {
        internal_node_data.push(MultilinearExtension::new(vec![p.clone()]));
        branch_data.push(MultilinearExtension::new(vec![if branch { Fr::one() } else { Fr::zero() }]));
        hash_data.push(MultilinearExtension::new(vec![next_hash.clone()]));
        let mut poseidon = poseidon::Poseidon::<Fr, T, RATE>::new(R_F, R_P);
        if !branch {
            poseidon.update(&vec![next_hash, p]);
        } else {
            poseidon.update(&vec![p, next_hash]);
        }
        next_hash = poseidon.state()[0];
    }
    let root_hash_data = MultilinearExtension::new(vec![next_hash]);
    (constants, mds_matrix, leaf_data, internal_node_data, branch_data, hash_data, root_hash_data)
}

fn main() {
    // For tracing.
    let _subscriber = fmt().with_max_level(Level::INFO).init();

    // Create the base layered circuit description.
    let circuit_compile_start = std::time::Instant::now();
    let base_circuit = build_circuit(NUM_LEAFS, PATH_LEN);
    let mut prover_circuit = base_circuit.clone();
    let verifier_circuit = base_circuit.clone();
    println!("Circuit build time: {} ms", circuit_compile_start.elapsed().as_millis());

    // Generate circuit inputs.
    let witness_gen_start = std::time::Instant::now();
    let leafs = gen_random_leafs(NUM_LEAFS);
    let paths = gen_random_path(PATH_LEN);
    let (constants, mds_matrix, leafs_data, internal_node_data, branch_data, _hash_data, root_hash_data) = gen_merkle_test(leafs, paths);
    println!("Witness gen time: {} ms", witness_gen_start.elapsed().as_millis());

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    let prove_start = std::time::Instant::now();
    for (i, consts) in constants.into_iter().enumerate() {
        let round_const_input_name = format!("Round constant {}", i);
        prover_circuit.set_input(&round_const_input_name, consts);
    }
    prover_circuit.set_input("MDS Matrix", mds_matrix);
    leafs_data.into_iter().enumerate().for_each(|(i, leaf_data)| {
        let leaf_input_name = format!("Leaf {}", i);
        prover_circuit.set_input(&leaf_input_name, leaf_data);
    });
    internal_node_data.into_iter().enumerate().for_each(|(i, internal_node_data)| {
        let internal_node_input_name = format!("Internal node {}", i);
        prover_circuit.set_input(&internal_node_input_name, internal_node_data);
    });
    branch_data.into_iter().enumerate().for_each(|(i, branch_data)| {
        let branch_input_name = format!("Branch {}", i);
        prover_circuit.set_input(&branch_input_name, branch_data);
    });
    prover_circuit.set_input("Expected root hash", root_hash_data); // This is public!

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let hyrax_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let hyrax_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&hyrax_circuit_prover_config, false);

    // Create a version of the circuit description which the prover can use.
    // Note that in this case, we create a "Hyrax-provable" circuit rather than
    // a "GKR-provable" one.
    let mut hyrax_provable_circuit: hyrax::provable_circuit::HyraxProvableCircuit<Bn256Point> =
        prover_circuit
            .gen_hyrax_provable_circuit()
            .expect("Failed to generate provable circuit");

    // The Pedersen committer creates and keeps track of the shared generators
    // between the prover and verifier. Note that the generators are created
    // deterministically from the public string.
    let prover_pedersen_committer =
        PedersenCommitter::new(512, "Hyrax tutorial Pedersen committer", None);

    // WARNING: This is for tutorial purposes ONLY. NEVER use anything but a CSPRNG for generating blinding factors!
    let mut blinding_rng = thread_rng();

    // The Vandermonde inverse matrix allows us to convert from evaluations
    // to coefficients for interpolative claim aggregation. Note that the
    // coefficient form allows the verifier to directly check relationships
    // via the homomorphic properties of the curve.
    let mut vandermonde_converter = VandermondeInverse::new();

    // Finally, we instantiate a transcript over the base field. Note that
    // prover messages are elliptic curve points which can be encoded as base
    // field tuples, while verifier messages are scalar field elements of that
    // curve. Thanks to Hasse's theorem, this results in a negligible completeness
    // loss in the non-interactive case as we always attempt to coerce a base
    // field challenge into a scalar field element and panic if the base field
    // element sampled was larger than the scalar field modulus.
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Fq>> =
        ECTranscript::new("Hyrax tutorial prover transcript");

    // Use the `perform_function_under_prover_config!` macro to run the
    // Hyrax prover's `prove` function with the above arguments, under the
    // prover config passed in.
    let (proof, proof_config) = perform_function_under_prover_config!(
        // This is a hack to get around the macro's syntax for struct methods
        // rather than function calls.
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z),
        &hyrax_circuit_prover_config,
        &prover_pedersen_committer,
        &mut blinding_rng,
        &mut vandermonde_converter,
        &mut prover_transcript
    );
    println!("Proof generation time: {} ms", prove_start.elapsed().as_millis());

    // ------------ VERIFIER ------------
    let verify_start = std::time::Instant::now();
    // We generate a "Hyrax-verifiable" circuit from the `Circuit` struct,
    // but we do not attach any circuit inputs to it (these must come from
    // the proof itself).
    let hyrax_verifiable_circuit = verifier_circuit
        .gen_hyrax_verifiable_circuit()
        .expect("Failed to generate Hyrax verifiable circuit");

    // The verifier can (and should) derive the elliptic curve generators on
    // its own from the public string and check the proof against these.
    let verifier_pedersen_committer =
        PedersenCommitter::new(512, "Hyrax tutorial Pedersen committer", None);

    // The verifier instantiates its own transcript.
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Fq>> =
        ECTranscript::new("Hyrax tutorial verifier transcript");

    // Finally, we verify the proof using the above committer + transcript, as
    // well as the Hyrax verifier config generated from the prover one earlier.
    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &hyrax_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &verifier_pedersen_committer,
        &mut verifier_transcript,
        &proof_config
    );
    println!("Proof verification time: {} ms", verify_start.elapsed().as_millis());

    let vc_size = bincode::serialized_size(&verifier_circuit).unwrap();
    let proof_size = bincode::serialized_size(&proof).unwrap();
    let proof_config_size = bincode::serialized_size(&proof_config).unwrap();
    let total_size = vc_size + proof_size + proof_config_size;
    println!("Total proof size: {} kb", total_size / 1024);
    println!("  verifiable circuit = {} kb\n  proof = {} kb", vc_size / 1024, proof_size / 1024);

    println!("All done! Hyrax proof generated and verified.");
}

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
const PATH_LEN: usize = 32; // number of poseidons
const NUM_VARS_PATH_LEN: usize = 5;

const R_F: usize = 8;
const R_P: usize = 57;
const NUM_VARS_T: usize = 2;
const T: usize = 3;
const RATE: usize = 2;

fn gen_random_leafs(num_leafs: usize) -> Vec<Fr> {
    (0..num_leafs).map(|_| Fr::random(thread_rng())).collect()
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
/// Use MDS matrix transpose so we can run matrix-vector product in parallel
fn full_poseidon(builder: &mut CircuitBuilder<Fr>, mut state: NodeRef<Fr>, round_consts: &Vec<NodeRef<Fr>>, mds_matrix_transpose: &NodeRef<Fr>) -> NodeRef<Fr> {
    // Full rounds
    for i in 0..R_F / 2 {
        // state += round_constants
        state = add_constants(builder, &state, round_consts, i);
        // state = state^5
        state = sbox_full(builder, &state);
        // state = MDS_matrix * state
        state = builder.add_matmult_node(
            &state,
            (NUM_VARS_PATH_LEN, NUM_VARS_T),
            mds_matrix_transpose,
            (NUM_VARS_T, NUM_VARS_T),
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
            &state,
            (NUM_VARS_PATH_LEN, NUM_VARS_T),
            mds_matrix_transpose,
            (NUM_VARS_T, NUM_VARS_T),
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
            &state,
            (NUM_VARS_PATH_LEN, NUM_VARS_T),
            mds_matrix_transpose,
            (NUM_VARS_T, NUM_VARS_T),
        );
    }
    state
}

/// number of poseidon in parallel, assume power of 2
fn build_circuit(num_poseidons: usize) -> Circuit<Fr> {
    assert!(num_poseidons.is_power_of_two());
    let log_2_num_poseidons = num_poseidons.trailing_zeros() as usize;
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
    let mds_matrix_transpose = builder.add_input_shred("MDS Matrix Transpose", 2 * NUM_VARS_T, &mds_layer);

    // Parallel poseidons
    // Multiple initial states in the same MLE
    let init_state = builder.add_input_shred("Init States", log_2_num_poseidons + NUM_VARS_T, &content_layer);

    // Generate root hash circuit
    let final_state = full_poseidon(&mut builder, init_state, &round_consts, &mds_matrix_transpose);

    // Output hash
    let expected_final_state =
        builder.add_input_shred("Expected final states", log_2_num_poseidons + NUM_VARS_T, &expected_hash_layer);
    let subtraction_sector = builder.add_sector(final_state - expected_final_state);
    builder.set_output(&subtraction_sector);

    builder.build().expect("Failed to build circuit")
}

// generate poseidon tests
fn gen_poseidon_test(num_poseidons: usize) -> (
    Vec<MultilinearExtension<Fr>>, // constants,
    MultilinearExtension<Fr>, // mds matrix,
    MultilinearExtension<Fr>, // init states
    MultilinearExtension<Fr>, // final states
) {
    let pad_len = T.next_power_of_two();
    // specs
    let spec = SpecRef::<Fr, T, RATE>::new(R_F, R_P);
    let (constants, mds_matrix) = (spec.constants(), spec.mds_matrices());
    let constants = constants.into_iter().map(|round_consts| {
        MultilinearExtension::new(round_consts.to_vec())
    }).collect();
    let mut mds_matrix_transpose = vec![Fr::zero(); pad_len * pad_len];
    for i in 0..T {
        for j in 0..T {
            mds_matrix_transpose[j * pad_len + i] = mds_matrix[i][j];
        }
    }
    let mds_matrix_transpose = MultilinearExtension::new(mds_matrix_transpose);
    // inputs / outputs
    let mut init_states = Vec::new();
    let mut final_states = Vec::new();
    for _ in 0..num_poseidons {
        let mut poseidon = poseidon::Poseidon::<Fr, T, RATE>::new(R_F, R_P);
        let mut init_state = gen_random_leafs(NUM_LEAFS);
        poseidon.update(&init_state);
        let mut final_state = poseidon.state().to_vec();
        init_state.insert(0, poseidon_iv());
        init_state.resize(pad_len, Fr::zero());
        final_state.resize(pad_len, Fr::zero());
        init_states.extend(init_state);
        final_states.extend(final_state);
    }
    let init_states = MultilinearExtension::new(init_states);
    let final_states = MultilinearExtension::new(final_states);
    (constants, mds_matrix_transpose, init_states, final_states)
}

fn main() {
    // For tracing.
    let _subscriber = fmt().with_max_level(Level::INFO).init();

    // Create the base layered circuit description.
    let circuit_compile_start = std::time::Instant::now();
    let base_circuit = build_circuit(PATH_LEN);
    let mut prover_circuit = base_circuit.clone();
    let verifier_circuit = base_circuit.clone();
    println!("Circuit build time: {} ms", circuit_compile_start.elapsed().as_millis());

    // Generate circuit inputs.
    let witness_gen_start = std::time::Instant::now();
    let (constants, mds_matrix_transpose, init_states, final_states) = gen_poseidon_test(PATH_LEN);
    println!("Witness gen time: {} ms", witness_gen_start.elapsed().as_millis());

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    let prove_start = std::time::Instant::now();
    for (i, consts) in constants.into_iter().enumerate() {
        let round_const_input_name = format!("Round constant {}", i);
        prover_circuit.set_input(&round_const_input_name, consts);
    }
    prover_circuit.set_input("MDS Matrix Transpose", mds_matrix_transpose);
    prover_circuit.set_input("Init States", init_states);
    prover_circuit.set_input("Expected final states", final_states);

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

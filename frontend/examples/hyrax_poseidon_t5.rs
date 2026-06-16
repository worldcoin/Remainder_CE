use std::iter;

use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use hyrax::gkr::verify_hyrax_proof;
use hyrax::utils::vandermonde::VandermondeInverse;
use rand::thread_rng;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig};
use shared_types::halo2curves::ff::PrimeField;
use shared_types::pedersen::PedersenCommitter;
use shared_types::transcript::ec_transcript::ECTranscript;
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::{
    ff_field, perform_function_under_prover_config, perform_function_under_verifier_config,
    Bn256Point, Fq, Fr,
};

use poseidon::{Spec, State};
use tracing::Level;
use tracing_subscriber::fmt;
use tracing_subscriber::{self};

const NUM_LEAFS: usize = 2;
const PATH_LEN: usize = 32; // number of poseidons
const NUM_VARS_PATH_LEN: usize = 5;

const R_F: usize = 8;
const R_P: usize = 57;
const HEAD_ROUNDS: usize = R_F / 2;
const PARTIAL_REPEAT: usize = 3;
const PARTIAL_ROUNDS: usize = R_P / PARTIAL_REPEAT;
const T: usize = 3;
const RATE: usize = 2;

fn gen_random_leafs(num_leafs: usize) -> Vec<Fr> {
    (0..num_leafs).map(|_| Fr::random(thread_rng())).collect()
}

// iv for halo2 poseidon
fn poseidon_iv() -> Fr {
    Fr::from_u128(1 << 64)
}

/// Add constants
fn add_constants(
    builder: &mut CircuitBuilder<Fr>,
    state: Vec<NodeRef<Fr>>,
    constants: &[Fr],
) -> Vec<NodeRef<Fr>> {
    state
        .into_iter()
        .zip(constants.into_iter())
        .map(|(s, c)| builder.add_sector(s + *c))
        .collect()
}
fn _add_constant(
    builder: &mut CircuitBuilder<Fr>,
    mut state: Vec<NodeRef<Fr>>,
    constant: &Fr,
) -> Vec<NodeRef<Fr>> {
    state[0] = builder.add_sector(state[0].clone() + *constant);
    state
}

/// Compute x^5
fn sbox(builder: &mut CircuitBuilder<Fr>, base: &NodeRef<Fr>) -> NodeRef<Fr> {
    builder.add_sector(base.clone() * base.clone() * base.clone() * base.clone() * base)
}
fn sbox_full(builder: &mut CircuitBuilder<Fr>, state: Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    state.iter().map(|s| sbox(builder, s)).collect()
}
fn _sbox_part(builder: &mut CircuitBuilder<Fr>, mut state: Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    state[0] = sbox(builder, &state[0]);
    state
}

/// Combinations
fn sbox_full_and_add_constants(builder: &mut CircuitBuilder<Fr>, state: Vec<NodeRef<Fr>>, constants: &[Fr]) -> Vec<NodeRef<Fr>> {
    state.into_iter().zip(constants.into_iter())
        .map(|(s, c)| builder.add_sector(s.clone() * s.clone() * s.clone() * s.clone() * s + *c))
        .collect()
}
fn sbox_part_and_add_constant(builder: &mut CircuitBuilder<Fr>, mut state: Vec<NodeRef<Fr>>, constant: &Fr) -> Vec<NodeRef<Fr>> {
    state[0] = builder.add_sector(state[0].clone() * state[0].clone() * state[0].clone() * state[0].clone() * state[0].clone() + *constant);
    state
}

/// Compute mds mat mult
fn mds(builder: &mut CircuitBuilder<Fr>, mds_matrix: &Vec<Vec<Fr>>, columns: &Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    mds_matrix.iter().map(|row| {
        let terms = row.iter().zip(columns.iter())
            .map(|(&coeff, col)| col.clone() * coeff)
            .reduce(|acc, term| acc + term)
            .unwrap_or_else(|| unreachable!());
        builder.add_sector(terms)
    }).collect()
}

fn mds_sparse(builder: &mut CircuitBuilder<Fr>, row: &[Fr; T], col_hat: &[Fr; RATE], state: Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    let old_first = state[0].clone();                 // OLD state[0]
    let new_first = {
        let term = row.iter().zip(state.iter())
            .map(|(&coeff, col)| col.clone() * coeff)
            .reduce(|acc, term| acc + term)
            .unwrap();
        builder.add_sector(term)
    };
    let mut new_state = Vec::with_capacity(T);
    new_state.push(new_first);
    for i in 1..T {
        new_state.push(builder.add_sector(old_first.clone() * col_hat[i - 1] + state[i].clone()));
    }
    new_state
}

/// One full poseidon
/// Use inputs by column
fn full_poseidon(
    builder: &mut CircuitBuilder<Fr>,
    mut head_state: Vec<NodeRef<Fr>>,
    mut partial_state: [Vec<NodeRef<Fr>>; PARTIAL_REPEAT],
    mut tail_state: Vec<NodeRef<Fr>>,
    spec: &Spec<Fr, T, RATE>,
) -> Vec<NodeRef<Fr>> {
    assert_eq!(R_P % PARTIAL_REPEAT, 0);
    // Record old states as internal states
    assert_eq!(head_state.len(), T - 1);
    for i in 0..PARTIAL_REPEAT {
        assert_eq!(partial_state[i].len(), T);
    }
    assert_eq!(tail_state.len(), T);
    let old_partial_state = partial_state.clone();
    let old_tail_state = tail_state.clone();

    // Prepend IV
    let iv = builder.add_sector(AbstractExpression::Constant(poseidon_iv()));
    head_state.insert(0, iv);

    // Head rounds
    head_state = add_constants(builder, head_state, &spec.constants().start()[0]);
    for round_constants in spec
        .constants()
        .start()
        .iter()
        .skip(1)
        .take(HEAD_ROUNDS - 1)
    {
        head_state = sbox_full_and_add_constants(builder, head_state, round_constants);
        head_state = mds(builder, &spec.mds_matrices().mds().as_vec(), &head_state);
    }
    head_state = sbox_full_and_add_constants(builder, head_state, spec.constants().start().last().unwrap());
    head_state = mds(
        builder,
        &spec.mds_matrices().pre_sparse_mds().as_vec(),
        &head_state,
    );

    // Partial rounds
    let partial_constants = spec.constants().partial();
    let sparse_matrices = spec.mds_matrices().sparse_matrices();
    for i in 0..PARTIAL_REPEAT {
        let start = i * PARTIAL_ROUNDS;
        for r in 0..PARTIAL_ROUNDS {
            partial_state[i] = sbox_part_and_add_constant(builder, partial_state[i].clone(), &partial_constants[start + r]);
            partial_state[i] = mds_sparse(
                builder,
                sparse_matrices[start + r].row(),
                sparse_matrices[start + r].col_hat(),
                partial_state[i].clone(),
            );
        }
    }
    // Tail rounds
    for round_constants in spec.constants().end().iter() {
        tail_state = sbox_full_and_add_constants(builder, tail_state, round_constants);
        tail_state = mds(builder, &spec.mds_matrices().mds().as_vec(), &tail_state);
    }
    tail_state = sbox_full(builder, tail_state);
    tail_state = mds(builder, &spec.mds_matrices().mds().as_vec(), &tail_state);

    // Assert internal states are the same
    for j in 0..T {
        let head_assert_sector =
            builder.add_sector(head_state[j].clone() - old_partial_state[0][j].clone());
        builder.set_output(&head_assert_sector);
    }
    for i in 0..PARTIAL_REPEAT - 1 {
        for j in 0..T {
            let partial_assert_sector = builder
                .add_sector(partial_state[i][j].clone() - old_partial_state[i + 1][j].clone());
            builder.set_output(&partial_assert_sector);
        }
    }
    for j in 0..T {
        let tail_assert_sector = builder
            .add_sector(partial_state[PARTIAL_REPEAT - 1][j].clone() - old_tail_state[j].clone());
        builder.set_output(&tail_assert_sector);
    }

    tail_state
}

/// number of poseidon in parallel, assume power of 2
fn build_circuit(num_poseidons: usize, spec: &Spec<Fr, T, RATE>) -> Circuit<Fr> {
    assert!(num_poseidons.is_power_of_two());
    let mut builder = CircuitBuilder::<Fr>::new();

    let content_layer = builder.add_input_layer("Content input layer", LayerVisibility::Committed);
    let expected_hash_layer = builder.add_input_layer("Expected hash", LayerVisibility::Public);

    // Parallel poseidons
    // Initial state grouped by poseidon entry (no IV)
    let head_state = (1..T)
        .map(|i| {
            let head_state_shred = builder.add_input_shred(
                &format!("Head state {}", i),
                NUM_VARS_PATH_LEN,
                &content_layer,
            );
            head_state_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();
    let partial_state = (0..PARTIAL_REPEAT)
        .map(|i| {
            (0..T)
                .map(|j| {
                    let partial_state_shred = builder.add_input_shred(
                        &format!("Partial state {}, {}", i, j),
                        NUM_VARS_PATH_LEN,
                        &content_layer,
                    );
                    partial_state_shred
                })
                .collect::<Vec<NodeRef<Fr>>>()
        })
        .collect::<Vec<Vec<NodeRef<Fr>>>>()
        .try_into()
        .unwrap();
    let tail_state = (0..T)
        .map(|i| {
            let final_state_shred = builder.add_input_shred(
                &format!("Tail state {}", i),
                NUM_VARS_PATH_LEN,
                &content_layer,
            );
            final_state_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();

    let final_state = full_poseidon(&mut builder, head_state, partial_state, tail_state, &spec);

    // Expected final state grouped by poseidon entry
    let expected_final_state = (0..T)
        .map(|i| {
            let expected_final_state_shred = builder.add_input_shred(
                &format!("Expected final state {}", i),
                NUM_VARS_PATH_LEN,
                &expected_hash_layer,
            );
            expected_final_state_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();

    // Output hash
    for (final_state, expected_final_state) in final_state
        .into_iter()
        .zip(expected_final_state.into_iter())
    {
        let subtraction_sector = builder.add_sector(final_state - expected_final_state);
        builder.set_output(&subtraction_sector);
    }

    builder
        .build_with_layer_combination()
        .expect("Failed to build circuit")
}

// generate poseidon tests
fn gen_poseidon_test(
    num_poseidons: usize,
    spec: &Spec<Fr, T, RATE>,
) -> (
    Vec<MultilinearExtension<Fr>>,      // head states, by column
    Vec<Vec<MultilinearExtension<Fr>>>, // partial states, by partial round and by column
    Vec<MultilinearExtension<Fr>>,      // tail states, by column
    Vec<MultilinearExtension<Fr>>,      // final states, by column
) {
    // inputs / outputs
    let mut head_states = vec![Vec::new(); T - 1];
    let mut partial_states = vec![vec![Vec::new(); T]; PARTIAL_REPEAT];
    let mut tail_states = vec![Vec::new(); T];
    let mut final_states = vec![Vec::new(); T];
    // rounds that we want to emit internal states
    let emit_rounds = iter::once(HEAD_ROUNDS - 1)
        .chain((0..PARTIAL_REPEAT).map(|i| HEAD_ROUNDS - 1 + PARTIAL_ROUNDS * (i + 1)))
        .collect::<Vec<usize>>();
    for _ in 0..num_poseidons {
        let mut state = State::default();
        let leafs = gen_random_leafs(NUM_LEAFS);
        state.add_init_state(&leafs);
        let emitted_states = spec.permute_and_emit_states(&mut state, &emit_rounds);

        // head_state are the leafs
        leafs
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| head_states[i].push(v));
        // partial states & tail states are the emitted internal states
        emitted_states
            .into_iter()
            .enumerate()
            .for_each(|(i, state)| {
                if i < PARTIAL_REPEAT {
                    state
                        .words()
                        .into_iter()
                        .enumerate()
                        .for_each(|(j, v)| partial_states[i][j].push(v));
                } else {
                    // tail rounds
                    state
                        .words()
                        .into_iter()
                        .enumerate()
                        .for_each(|(j, v)| tail_states[j].push(v));
                }
            });
        // final state is state
        state
            .words()
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| final_states[i].push(v));
    }
    let head_states: Vec<MultilinearExtension<Fr>> = head_states
        .into_iter()
        .map(|head_state| MultilinearExtension::new(head_state))
        .collect();
    let partial_states: Vec<Vec<MultilinearExtension<Fr>>> = partial_states
        .into_iter()
        .map(|partial_state| {
            partial_state
                .into_iter()
                .map(|state| MultilinearExtension::new(state))
                .collect()
        })
        .collect();
    let tail_states: Vec<MultilinearExtension<Fr>> = tail_states
        .into_iter()
        .map(|tail_state| MultilinearExtension::new(tail_state))
        .collect();
    let final_states: Vec<MultilinearExtension<Fr>> = final_states
        .into_iter()
        .map(|final_state| MultilinearExtension::new(final_state))
        .collect();
    (head_states, partial_states, tail_states, final_states)
}

fn main() {
    // For tracing.
    let _subscriber = fmt().with_max_level(Level::INFO).init();

    // Create the base layered circuit description.
    let circuit_compile_start = std::time::Instant::now();
    let spec = Spec::<Fr, T, RATE>::new(R_F, R_P);
    let base_circuit = build_circuit(PATH_LEN, &spec);
    let mut prover_circuit = base_circuit.clone();
    let verifier_circuit = base_circuit.clone();
    println!(
        "Circuit build time: {} ms",
        circuit_compile_start.elapsed().as_millis()
    );

    // Generate circuit inputs.
    let witness_gen_start = std::time::Instant::now();
    let (head_states, partial_states, tail_states, final_states) =
        gen_poseidon_test(PATH_LEN, &spec);
    println!(
        "Witness gen time: {} ms",
        witness_gen_start.elapsed().as_millis()
    );

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    let prove_start = std::time::Instant::now();
    for (i, head_state) in head_states.into_iter().enumerate() {
        let head_state_input_name = format!("Head state {}", i + 1);
        prover_circuit.set_input(&head_state_input_name, head_state);
    }
    for (i, partial_states) in partial_states.into_iter().enumerate() {
        for (j, partial_state) in partial_states.into_iter().enumerate() {
            let partial_state_input_name = format!("Partial state {}, {}", i, j);
            prover_circuit.set_input(&partial_state_input_name, partial_state);
        }
    }
    for (i, tail_state) in tail_states.into_iter().enumerate() {
        let tail_state_input_name = format!("Tail state {}", i);
        prover_circuit.set_input(&tail_state_input_name, tail_state);
    }
    for (i, expected_final_state) in final_states.into_iter().enumerate() {
        let expected_final_state_input_name = format!("Expected final state {}", i);
        prover_circuit.set_input(&expected_final_state_input_name, expected_final_state);
    }

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
    println!(
        "Proof generation time: {} ms",
        prove_start.elapsed().as_millis()
    );

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
    println!(
        "Proof verification time: {} ms",
        verify_start.elapsed().as_millis()
    );

    let vc_size = bincode::serialized_size(&verifier_circuit).unwrap();
    let proof_size = bincode::serialized_size(&proof).unwrap();
    let proof_config_size = bincode::serialized_size(&proof_config).unwrap();
    let total_size = vc_size + proof_size + proof_config_size;
    println!("Total proof size: {} kb", total_size / 1024);
    println!(
        "  verifiable circuit = {} kb\n  proof = {} kb",
        vc_size / 1024,
        proof_size / 1024
    );

    println!("All done! Hyrax proof generated and verified.");
}

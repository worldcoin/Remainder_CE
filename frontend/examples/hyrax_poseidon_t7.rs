use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef};
use hyrax::gkr::{HyraxProof, verify_hyrax_proof};
use hyrax::utils::vandermonde::VandermondeInverse;
use rand::thread_rng;
use remainder::mle::evals::MultilinearExtension;
use shared_types::config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig};
use shared_types::curves::PrimeOrderCurve;
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
const NUM_INTERMEDIATE_STATES: usize = 4; // excluding leafs and final state
const T: usize = 3;
const RATE: usize = 2;

fn get_round_cutoffs() -> Vec<usize> {
    (0..NUM_INTERMEDIATE_STATES)
        .map(|i| (i + 1) * (R_F + R_P) / (NUM_INTERMEDIATE_STATES + 1))
        .collect::<Vec<usize>>()
}

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
    let mult_int = builder.add_sector(base.clone() * base.clone());
    builder.add_sector(mult_int.clone() * mult_int * base)
}
fn sbox_full(builder: &mut CircuitBuilder<Fr>, state: Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    state.iter().map(|s| sbox(builder, s)).collect()
}
fn _sbox_part(builder: &mut CircuitBuilder<Fr>, mut state: Vec<NodeRef<Fr>>) -> Vec<NodeRef<Fr>> {
    state[0] = sbox(builder, &state[0]);
    state
}

/// Combinations
fn sbox_full_and_add_constants(
    builder: &mut CircuitBuilder<Fr>,
    state: Vec<NodeRef<Fr>>,
    constants: &[Fr],
) -> Vec<NodeRef<Fr>> {
    state
        .into_iter()
        .zip(constants.into_iter())
        .map(|(s, c)| {
            let mult_int = builder.add_sector(s.clone() * s.clone());
            builder.add_sector(mult_int.clone() * mult_int * s + *c)
        })
        .collect()
}
fn sbox_part_and_add_constant(
    builder: &mut CircuitBuilder<Fr>,
    mut state: Vec<NodeRef<Fr>>,
    constant: &Fr,
) -> Vec<NodeRef<Fr>> {
    let mult_int = builder.add_sector(state[0].clone() * state[0].clone());
    state[0] = builder.add_sector(
        mult_int.clone()
            * mult_int
            * state[0].clone()
            + *constant,
    );
    state
}

/// Compute mds mat mult
fn mds(
    builder: &mut CircuitBuilder<Fr>,
    mds_matrix: &Vec<Vec<Fr>>,
    columns: &Vec<NodeRef<Fr>>,
) -> Vec<NodeRef<Fr>> {
    mds_matrix
        .iter()
        .map(|row| {
            let terms = row
                .iter()
                .zip(columns.iter())
                .map(|(&coeff, col)| col.clone() * coeff)
                .reduce(|acc, term| acc + term)
                .unwrap_or_else(|| unreachable!());
            builder.add_sector(terms)
        })
        .collect()
}
fn mds_sparse(
    builder: &mut CircuitBuilder<Fr>,
    row: &[Fr; T],
    col_hat: &[Fr; RATE],
    state: Vec<NodeRef<Fr>>,
) -> Vec<NodeRef<Fr>> {
    let old_first = state[0].clone(); // OLD state[0]
    let new_first = {
        let term = row
            .iter()
            .zip(state.iter())
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

/// Execute one Poseidon round given its ABSOLUTE index `i` in [0, R_F + R_P).
/// Everything (full vs partial, which constant, which matrix) is a function of
/// `i` alone, so segments are just contiguous index ranges.
fn run_round(
    builder: &mut CircuitBuilder<Fr>,
    mut state: Vec<NodeRef<Fr>>,
    i: usize,
    spec: &Spec<Fr, T, RATE>,
    mds_vec: &Vec<Vec<Fr>>,
    pre_sparse_vec: &Vec<Vec<Fr>>,
) -> Vec<NodeRef<Fr>> {
    let head = R_F / 2; // head rounds: 0 .. head
    let partial_end = head + R_P; // partial rounds: head .. partial_end
    let total = R_F + R_P;

    // Folded round-0 constant (start[0]) added before the first sbox.
    // Only fires in the leafs segment, since round 0 must live there.
    if i == 0 {
        state = add_constants(builder, state, &spec.constants().start()[0]);
    }

    if i < head {
        // Head full round: sbox -> add start[i+1] -> mds (last head round uses pre_sparse)
        state = sbox_full_and_add_constants(builder, state, &spec.constants().start()[i + 1]);
        let matrix = if i == head - 1 {
            pre_sparse_vec
        } else {
            mds_vec
        };
        state = mds(builder, matrix, &state);
    } else if i < partial_end {
        // Partial round
        let k = i - head;
        state = sbox_part_and_add_constant(builder, state, &spec.constants().partial()[k]);
        let sparse = &spec.mds_matrices().sparse_matrices()[k];
        state = mds_sparse(builder, sparse.row(), sparse.col_hat(), state);
    } else {
        // End full round: the final round (i == total-1) adds no constant
        let k = i - partial_end;
        if i < total - 1 {
            state = sbox_full_and_add_constants(builder, state, &spec.constants().end()[k]);
        } else {
            state = sbox_full(builder, state);
        }
        state = mds(builder, mds_vec, &state);
    }
    state
}

/// One full poseidon
/// Use inputs by column
fn full_poseidon(
    builder: &mut CircuitBuilder<Fr>,
    mut leafs: Vec<NodeRef<Fr>>,
    intermediate_state: [Vec<NodeRef<Fr>>; NUM_INTERMEDIATE_STATES],
    spec: &Spec<Fr, T, RATE>,
) -> Vec<NodeRef<Fr>> {
    let total = R_F + R_P;
    let cutoffs = get_round_cutoffs();

    // Hoist the dense matrices once rather than reallocating per round.
    let mds_vec = spec.mds_matrices().mds().as_vec();
    let pre_sparse_vec = spec.mds_matrices().pre_sparse_mds().as_vec();

    // --- Leafs segment: prepend IV, run rounds [0, cutoffs[0]) ---
    assert_eq!(leafs.len(), T - 1);
    let leafs_end = cutoffs.first().copied().unwrap_or(total);
    assert!(
        leafs_end >= 1,
        "round 0 (IV + start[0]) must fall in the leafs segment"
    );
    let iv = builder.add_sector(AbstractExpression::Constant(poseidon_iv()));
    leafs.insert(0, iv);
    for i in 0..leafs_end {
        leafs = run_round(builder, leafs, i, spec, &mds_vec, &pre_sparse_vec);
    }
    let mut new_intermediate_state = vec![leafs];

    // --- Intermediate segments: state j runs rounds [cutoffs[j], cutoffs[j+1] | total) ---
    for (j, mut state) in intermediate_state.clone().into_iter().enumerate() {
        assert_eq!(state.len(), T);
        let start = cutoffs[j];
        let end = if j + 1 < NUM_INTERMEDIATE_STATES {
            cutoffs[j + 1]
        } else {
            total
        };
        for i in start..end {
            state = run_round(builder, state, i, spec, &mds_vec, &pre_sparse_vec);
        }
        new_intermediate_state.push(state);
    }

    // Assert internal states are the same
    for i in 0..NUM_INTERMEDIATE_STATES {
        for j in 0..T {
            let intermediate_assert_sector = builder.add_sector(
                new_intermediate_state[i][j].clone() - intermediate_state[i][j].clone(),
            );
            builder.set_output(&intermediate_assert_sector);
        }
    }

    // Permutation output = the last intermediate
    new_intermediate_state.pop().unwrap()
}

/// number of poseidon in parallel, assume power of 2
fn build_circuit(num_poseidons: usize, spec: &Spec<Fr, T, RATE>) -> Circuit<Fr> {
    assert!(num_poseidons.is_power_of_two());
    let mut builder = CircuitBuilder::<Fr>::new();

    let content_layer = builder.add_input_layer("Content input layer", LayerVisibility::Committed);
    let expected_hash_layer = builder.add_input_layer("Expected hash", LayerVisibility::Public);

    // Parallel poseidons
    // Initial state grouped by poseidon entry (no IV)
    let leafs = (1..T)
        .map(|i| {
            let leaf_shred = builder.add_input_shred(
                &format!("Leaf state {}", i),
                NUM_VARS_PATH_LEN,
                &content_layer,
            );
            leaf_shred
        })
        .collect::<Vec<NodeRef<Fr>>>();
    let intermediate_state = (0..NUM_INTERMEDIATE_STATES)
        .map(|i| {
            (0..T)
                .map(|j| {
                    let intermediate_state_shred = builder.add_input_shred(
                        &format!("Intermediate state {}, {}", i, j),
                        NUM_VARS_PATH_LEN,
                        &content_layer,
                    );
                    intermediate_state_shred
                })
                .collect::<Vec<NodeRef<Fr>>>()
        })
        .collect::<Vec<Vec<NodeRef<Fr>>>>()
        .try_into()
        .unwrap();
    let final_state = full_poseidon(&mut builder, leafs, intermediate_state, &spec);

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
    Vec<Vec<MultilinearExtension<Fr>>>, // intermediate states, by intermediate round and by column
    Vec<MultilinearExtension<Fr>>,      // final states, by column
) {
    // inputs / outputs
    let mut leaf_state = vec![Vec::new(); T - 1];
    let mut intermediate_states = vec![vec![Vec::new(); T]; NUM_INTERMEDIATE_STATES];
    let mut final_states = vec![Vec::new(); T];
    // rounds that we want to emit internal states
    let emit_rounds = get_round_cutoffs()
        .into_iter()
        .map(|c| c - 1)
        .collect::<Vec<usize>>();
    for _ in 0..num_poseidons {
        let mut state = State::default();
        let leafs = gen_random_leafs(NUM_LEAFS);
        state.add_init_state(&leafs);
        let emitted_states = spec.permute_and_emit_states(&mut state, &emit_rounds);

        // leaf states are the leafs
        leafs
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| leaf_state[i].push(v));
        // intermediate states are the emitted internal states
        emitted_states
            .into_iter()
            .enumerate()
            .for_each(|(i, state)| {
                state
                    .words()
                    .into_iter()
                    .enumerate()
                    .for_each(|(j, v)| intermediate_states[i][j].push(v));
            });
        // final state is state
        state
            .words()
            .into_iter()
            .enumerate()
            .for_each(|(i, v)| final_states[i].push(v));
    }

    let leaf_states: Vec<MultilinearExtension<Fr>> = leaf_state
        .into_iter()
        .map(|leaf_state| MultilinearExtension::new(leaf_state))
        .collect();
    let intermediate_states: Vec<Vec<MultilinearExtension<Fr>>> = intermediate_states
        .into_iter()
        .map(|intermediate_state| {
            intermediate_state
                .into_iter()
                .map(|state| MultilinearExtension::new(state))
                .collect()
        })
        .collect();
    let final_states: Vec<MultilinearExtension<Fr>> = final_states
        .into_iter()
        .map(|final_state| MultilinearExtension::new(final_state))
        .collect();
    (leaf_states, intermediate_states, final_states)
}

pub fn print_proof_size<C: PrimeOrderCurve>(proof: &HyraxProof<C>) {
    let public_inputs_size = bincode::serialize(&proof.public_inputs).unwrap().len();
    println!("    public inputs = {} kb", public_inputs_size / 1024);
    let circuit_proof_size = bincode::serialize(&proof.circuit_proof).unwrap().len();
    println!("    circuit proof = {} kb", circuit_proof_size / 1024);
    {
        let layer_proof_size = bincode::serialize(&proof.circuit_proof.layer_proofs).unwrap().len();
        println!("      layer proofs = {} kb, {} proofs", layer_proof_size / 1024, proof.circuit_proof.layer_proofs.len());
        {
            let total_sumcheck_proof_size = proof.circuit_proof.layer_proofs.iter().map(|p| 
                bincode::serialize(&p.1.proof_of_sumcheck).unwrap().len()
            ).sum::<usize>();
            println!("        total sumcheck proof size = {} kb", total_sumcheck_proof_size / 1024);
            let total_commitments_size = proof.circuit_proof.layer_proofs.iter().map(|p| 
                bincode::serialize(&p.1.commitments).unwrap().len()
            ).sum::<usize>();
            println!("        total commitments size = {} kb", total_commitments_size / 1024);
            let total_proofs_of_product_count = proof.circuit_proof.layer_proofs.iter().map(|p| 
                p.1.proofs_of_product.len()
            ).sum::<usize>();
            let total_proofs_of_product_size = proof.circuit_proof.layer_proofs.iter().map(|p| 
                bincode::serialize(&p.1.proofs_of_product).unwrap().len()
            ).sum::<usize>();
            println!("        total proofs of product size = {} kb, {} proofs", total_proofs_of_product_size / 1024, total_proofs_of_product_count);
            let total_maybe_proof_of_claim_agg_size = proof.circuit_proof.layer_proofs.iter().map(|p| 
                p.1.maybe_proof_of_claim_agg.as_ref().map(|proof_of_claim_agg| bincode::serialize(proof_of_claim_agg).unwrap().len()).unwrap_or(0)
            ).sum::<usize>();
            println!("        total maybe proof of claim agg size = {} kb", total_maybe_proof_of_claim_agg_size / 1024);
        }
        let output_layer_proof_size = bincode::serialize(&proof.circuit_proof.output_layer_proofs).unwrap().len();
        println!("      output layer proof = {} kb", output_layer_proof_size / 1024);
        let fiat_shamir_claim_size = bincode::serialize(&proof.circuit_proof.fiat_shamir_claims).unwrap().len();
        println!("      Fiat-Shamir claims = {} kb", fiat_shamir_claim_size / 1024);
    }
    let claims_size = bincode::serialize(&proof.claims_on_public_values)
        .unwrap()
        .len();
    println!("    claims on public values = {} kb", claims_size / 1024);
    let hyrax_input_proofs = bincode::serialize(&proof.hyrax_input_proofs).unwrap().len();
    println!("    hyrax input proofs = {} kb", hyrax_input_proofs / 1024);
    
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
    let (leaf_states, intermediate_states, final_states) = gen_poseidon_test(PATH_LEN, &spec);
    println!(
        "Witness gen time: {} ms",
        witness_gen_start.elapsed().as_millis()
    );

    // Append circuit inputs to their respective input "shreds" in the prover's
    // view of the circuit.
    let prove_start = std::time::Instant::now();
    for (i, leaf_state) in leaf_states.into_iter().enumerate() {
        let leaf_state_input_name = format!("Leaf state {}", i + 1);
        prover_circuit.set_input(&leaf_state_input_name, leaf_state);
    }
    for (i, intermediate_states) in intermediate_states.into_iter().enumerate() {
        for (j, intermediate_state) in intermediate_states.into_iter().enumerate() {
            let intermediate_state_input_name = format!("Intermediate state {}, {}", i, j);
            prover_circuit.set_input(&intermediate_state_input_name, intermediate_state);
        }
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
    print_proof_size(&proof);

    println!("All done! Hyrax proof generated and verified.");
}

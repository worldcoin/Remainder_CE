use std::path::Path;

use crate::hyrax_gkr::hyrax_input_layer::InputProofEnum;
use crate::hyrax_gkr::hyrax_output_layer::HyraxOutputLayer;
use crate::hyrax_gkr::{Circuit, CircuitDescription};
use crate::hyrax_pcs::MleCoefficientsVector;
use crate::layer::matmult::Matrix;
use crate::layer::LayerId;
use crate::logup::circuits::{fractional_sumcheck, setup_logup_mles};
use crate::mle::dense::DenseMle;
use crate::mle::{Mle, MleRef};
use crate::pedersen::PedersenCommitter;
use crate::prover::input_layer::combine_input_layers::InputLayerBuilder;
use crate::prover::input_layer::enum_input_layer::{CommitmentEnum, InputLayerEnum};
use crate::prover::input_layer::hyrax_input_layer::HyraxInputLayer;
use crate::prover::input_layer::ligero_input_layer::LigeroInputLayer;
use crate::prover::input_layer::public_input_layer::PublicInputLayer;
use crate::prover::input_layer::random_input_layer::RandomInputLayer;
use crate::prover::input_layer::InputLayer;
use crate::utils::vandermonde::VandermondeInverse;
use crate::worldcoin::builders::SignedRecompBuilder;
use crate::worldcoin::data::{load_data, load_data_from_serialized_inputs, WorldcoinData};
use crate::worldcoin::digit_decomposition::{DigitRecompBuilder, BASE};
use crate::{curves::PrimeOrderCurve, prover::GKRCircuit, worldcoin::data::WorldcoinCircuitData};
use ark_std::{end_timer, log2, start_timer};
use ark_test_curves::bn::Bn;
use halo2_base::halo2_proofs::poly::commitment;
use itertools::Itertools;
use remainder_shared_types::transcript::poseidon_transcript::PoseidonTranscript;
use remainder_shared_types::transcript::Transcript;
use remainder_shared_types::ScalarField;
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    transcript::counting_transcript::CountingTranscript,
    FieldExt, Poseidon,
};

use super::HyraxProof;
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

pub struct WorldcoinCircuitPrecommit<P: PrimeOrderCurve> {
    pub worldcoin_circuit_data: WorldcoinCircuitData<Scalar>,
    pub hyrax_precommit: Vec<P>,
    pub blinding_factors_matrix: Vec<Scalar>,
    pub log_num_cols: usize,
}

fn generate_hyrax_WC_circuit(
    worldcoin_precommit_data: WorldcoinCircuitPrecommit<Bn256Point>,
    prover_transcript: &mut PoseidonTranscript<Scalar, Base>,
) -> Circuit<Bn256Point, PoseidonTranscript<Scalar, Base>> {
    type Transcript = PoseidonTranscript<Scalar, Base>;

    const HYRAX_INPUT_LAYER_ID: usize = 0;
    const DIGIT_DECOMP_LAYER_ID: usize = 1;
    const PUBLIC_INPUT_LAYER_ID: usize = 2;
    const RANDOM_LAYER_ID: usize = 3;
    const INVERSES_LAYER_ID: usize = 4;

    let WorldcoinCircuitData {
        mut image_matrix_mle,
        reroutings: wirings,
        num_placements,
        mut kernel_matrix_mle,
        kernel_matrix_dims,
        mut digits,
        mut iris_code,
        mut digit_multiplicities,
    } = worldcoin_precommit_data.worldcoin_circuit_data.clone();

    // INPUT LAYER
    let mut input_layers = vec![];
    let mut input_commitments = vec![];

    // --- Hyrax input commit ---
    image_matrix_mle.layer_id = LayerId::Input(HYRAX_INPUT_LAYER_ID);

    let image_matrix_mle_vec_u8 = image_matrix_mle
        .mle_ref()
        .bookkeeping_table
        .iter()
        .map(|elem| {
            let elem_bytes = elem.to_bytes_le();
            let only_sig_bit = elem_bytes[0];
            only_sig_bit
        })
        .collect_vec();
    let mle_coefficients_vec = MleCoefficientsVector::U8Vector(image_matrix_mle_vec_u8);

    let committer: PedersenCommitter<Bn256Point> = PedersenCommitter::new(
        (1 << worldcoin_precommit_data.log_num_cols) + 1,
        "Modulus <3 Worldcoin: ZKML Self-Custody Edition",
        Some(8),
    );

    let image_input_layer = HyraxInputLayer::<Bn256Point, Transcript>::new_with_hyrax_commitment(
        mle_coefficients_vec,
        LayerId::Input(HYRAX_INPUT_LAYER_ID),
        committer.clone(),
        worldcoin_precommit_data.blinding_factors_matrix.clone(),
        worldcoin_precommit_data.log_num_cols,
        worldcoin_precommit_data.hyrax_precommit.clone(),
    );

    let image_input_layer = image_input_layer.to_enum();

    let hyrax_precommit =
        CommitmentEnum::HyraxCommitment(worldcoin_precommit_data.hyrax_precommit.clone());
    InputLayerEnum::append_commitment_to_transcript(&hyrax_precommit, prover_transcript);

    input_layers.push(image_input_layer);
    input_commitments.push(hyrax_precommit);

    // --- Ligero input commit ---
    digits.layer_id = LayerId::Input(DIGIT_DECOMP_LAYER_ID);
    digit_multiplicities.layer_id = LayerId::Input(DIGIT_DECOMP_LAYER_ID);
    let digit_multiplicities_vec: Vec<Box<&mut dyn Mle<Scalar>>> =
        vec![Box::new(&mut digits), Box::new(&mut digit_multiplicities)];
    let digit_mult_mles_builder = InputLayerBuilder::<Bn256Point>::new(
        digit_multiplicities_vec,
        None,
        LayerId::Input(DIGIT_DECOMP_LAYER_ID),
    );

    let digit_input_layer: HyraxInputLayer<Bn256Point, Transcript> =
        digit_mult_mles_builder.to_input_layer::<HyraxInputLayer<Bn256Point, Transcript>>();
    let mut digit_input_layer = digit_input_layer.to_enum();
    let digit_commit = digit_input_layer.commit().unwrap();
    InputLayerEnum::append_commitment_to_transcript(&digit_commit, prover_transcript);

    input_layers.push(digit_input_layer);
    input_commitments.push(digit_commit);

    // --- Public inputs (at least: those already available) ---
    kernel_matrix_mle.layer_id = LayerId::Input(PUBLIC_INPUT_LAYER_ID);
    iris_code.layer_id = LayerId::Input(PUBLIC_INPUT_LAYER_ID);
    let mut digit_lookup_table = DenseMle::new_from_iter(
        (0u64..BASE as u64).into_iter().map(|x| Scalar::from(x)),
        LayerId::Input(PUBLIC_INPUT_LAYER_ID),
        None,
    );

    let public_input_mles: Vec<Box<&mut dyn Mle<Scalar>>> = vec![
        Box::new(&mut kernel_matrix_mle),
        Box::new(&mut iris_code),
        Box::new(&mut digit_lookup_table),
    ];

    let public_input_mles_builder = InputLayerBuilder::<Bn256Point>::new(
        public_input_mles,
        None,
        LayerId::Input(PUBLIC_INPUT_LAYER_ID),
    );

    let public_input_layer: PublicInputLayer<Bn256Point, Scalar, Transcript> =
        public_input_mles_builder
            .to_input_layer::<PublicInputLayer<Bn256Point, Scalar, Transcript>>();
    let mut public_input_layer = public_input_layer.to_enum();
    let public_commit = public_input_layer.commit().unwrap();
    InputLayerEnum::append_commitment_to_transcript(&public_commit, prover_transcript);

    input_layers.push(public_input_layer);
    input_commitments.push(public_commit);

    // Start LogUp:
    // NB we should just be able to call a single function, but I haven't been able to find a clean interface

    // --- Random Input Layer ---
    // Generate the SZ evaluation point using Fiat-Shamir
    let random_layer: RandomInputLayer<_, _, _, Transcript> =
        RandomInputLayer::new(prover_transcript, 1, LayerId::Input(RANDOM_LAYER_ID));
    let r_mle = random_layer.get_mle();
    let mut random_layer_enum: InputLayerEnum<Bn256Point, Transcript> = random_layer.to_enum();
    let random_commit = random_layer_enum.commit().unwrap();

    input_layers.push(random_layer_enum);
    input_commitments.push(random_commit);

    let (
        witness_numerators,
        witness_denominators,
        table_denominators,
        witness_sum_denom_inverse,
        table_sum_denom_inverse,
        mut layers,
    ) = setup_logup_mles(
        r_mle.clone(),
        digits.get_entire_mle_as_mle_ref(),
        digit_lookup_table.mle_ref(),
    );

    let mut witness_sum_denom_inverse = DenseMle::new_from_iter(
        vec![witness_sum_denom_inverse].into_iter(),
        LayerId::Input(INVERSES_LAYER_ID),
        None,
    );
    let mut table_sum_denom_inverse = DenseMle::new_from_iter(
        vec![table_sum_denom_inverse].into_iter(),
        LayerId::Input(INVERSES_LAYER_ID),
        None,
    );

    // --- Inverses Input Layer ---
    let inverses_mles: Vec<Box<&mut dyn Mle<Scalar>>> = vec![
        Box::new(&mut witness_sum_denom_inverse),
        Box::new(&mut table_sum_denom_inverse),
    ];

    let inverses_il_builder = InputLayerBuilder::<Bn256Point>::new(
        inverses_mles,
        None,
        LayerId::Input(INVERSES_LAYER_ID),
    );

    let inverses_input_layer: HyraxInputLayer<Bn256Point, Transcript> =
        inverses_il_builder.to_input_layer::<HyraxInputLayer<Bn256Point, Transcript>>();
    let mut inverses_input_layer = inverses_input_layer.to_enum();
    let inverses_commit = inverses_input_layer.commit().unwrap();
    InputLayerEnum::append_commitment_to_transcript(&inverses_commit, prover_transcript);

    input_layers.push(inverses_input_layer);
    input_commitments.push(inverses_commit);

    let mut output_layers = fractional_sumcheck(
        witness_numerators,
        witness_denominators,
        witness_sum_denom_inverse,
        digit_multiplicities,
        table_denominators,
        table_sum_denom_inverse,
        &mut layers,
    );

    // End LogUp

    let (filter_num_rows, filter_num_cols) = kernel_matrix_dims;
    let (filter_num_rows_vars, filter_num_cols_vars) = (
        log2(filter_num_rows) as usize,
        log2(filter_num_cols) as usize,
    );

    // "Re-routing" layer that transforms the input image into a matrix A
    let matrix_a_mle = layers.add_identity_gate(wirings, image_matrix_mle.mle_ref());
    let matrix_a = Matrix::new(
        matrix_a_mle.mle_ref(),
        log2(num_placements) as usize,
        filter_num_rows_vars,
    );

    // Form the matrix B consisting of filter values
    let matrix_b = Matrix::new(
        kernel_matrix_mle.mle_ref(),
        filter_num_rows_vars,
        filter_num_cols_vars,
    );

    // Add the matrix multiplication layer
    // Resulting values are equal to value of convolutional filter (from matrix_b) at a placement (from matrix_a)
    let result_of_matmult = layers.add_matmult_layer(matrix_a, matrix_b);

    // Layer to check the digital recomposition of the responses
    let digital_recomp_builder = DigitRecompBuilder {
        unsigned_digit_decomp: digits,
        base: BASE as u64,
    };
    let recomp_of_abs_value = layers.add_gkr(digital_recomp_builder);

    // Layer to check that sign bits (=iris code) combined with the recomp of the absolute value gives the result of the matrix multiplication
    let recomp_check_builder =
        SignedRecompBuilder::new(result_of_matmult, iris_code, recomp_of_abs_value);
    output_layers.push(layers.add_gkr(recomp_check_builder).get_enum());

    let output_layers = output_layers
        .into_iter()
        .map(|output_layer| HyraxOutputLayer {
            underlying_mle: output_layer,
        })
        .collect_vec();

    let layers = layers.0;
    let circuit = Circuit {
        input_layers,
        layers,
        output_layers,
        input_commitments: input_commitments.clone(),
    };

    circuit
}

#[test]
fn test_WC_circuit_full_hyrax() {
    let mut blinding_rng = &mut rand::thread_rng();
    type Transcript = PoseidonTranscript<Scalar, Base>;
    let prover_transcript = &mut Transcript::new("");
    let verifier_transcript = &mut Transcript::new("");
    const NUM_GENERATORS: usize = 512;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "Modulus <3 Worldcoin: ZKML Self-Custody Edition",
        None,
    );

    let (commitment, blinding_factors, data): (
        Vec<Bn256Point>,
        Vec<Scalar>,
        WorldcoinData<Scalar>,
    ) = load_data_from_serialized_inputs::<Bn256Point>(
        "worldcoin_witness_data/right_normalized_image_resized.bin",
        "worldcoin_witness_data/right_normalized_image_blinding_factors_resized.bin",
        "worldcoin_witness_data/right_normalized_image_commitment_resized.bin",
        "worldcoin_witness_data/iris_codes.json",
        "right_iris_code",
        Path::new("worldcoin_witness_data").to_path_buf(),
        (100) as usize,
        (400) as usize,
    );
    let worldcoin_circuit_data = (&data).into();

    let worldcoin_circuit_with_precommit: WorldcoinCircuitPrecommit<Bn256Point> =
        WorldcoinCircuitPrecommit {
            worldcoin_circuit_data,
            hyrax_precommit: commitment,
            blinding_factors_matrix: blinding_factors,
            log_num_cols: 9,
        };

    let mut wc_circuit =
        generate_hyrax_WC_circuit(worldcoin_circuit_with_precommit, prover_transcript);

    let prove_timer = start_timer!(|| "prove timer");
    // PROVE
    let proof = HyraxProof::prove(
        &mut wc_circuit,
        &committer,
        &mut blinding_rng,
        prover_transcript,
        &mut VandermondeInverse::new(),
    );
    end_timer!(prove_timer);

    // VERIFY
    let circuit_description: CircuitDescription<Bn256Point> = wc_circuit.into();

    // Add input layer commitments to transcript first
    proof
        .input_layer_proofs
        .iter()
        .for_each(|input_layer_proof| match input_layer_proof {
            InputProofEnum::HyraxInputLayerProof(hyrax_input_proof) => {
                InputLayerEnum::append_commitment_to_transcript(
                    &CommitmentEnum::HyraxCommitment(hyrax_input_proof.input_commitment.clone()),
                    verifier_transcript,
                )
                .unwrap();
            }
            // @vishady hope I'm doing this bit correctly
            InputProofEnum::PublicInputLayerProof(layer, _) => {
                PublicInputLayer::<Bn256Point, Scalar, _>::append_commitment_to_transcript(
                    &layer.clone().commit().unwrap(),
                    verifier_transcript,
                )
                .unwrap();
            }
            InputProofEnum::RandomInputLayerProof(layer, _) => {
                RandomInputLayer::<Bn256Point, Scalar, Base, _>::append_commitment_to_transcript(
                    &layer.clone().commit().unwrap(),
                    verifier_transcript,
                )
                .unwrap();
            }
        });

    let verify_timer = start_timer!(|| "proof verification");
    HyraxProof::verify(
        &proof,
        &circuit_description,
        &committer,
        verifier_transcript,
    );
    end_timer!(verify_timer);
}

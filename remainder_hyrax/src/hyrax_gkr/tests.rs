use crate::hyrax_gkr::hyrax_input_layer::HyraxInputLayerProof;
use crate::hyrax_gkr::hyrax_layer::HyraxClaim;
use crate::hyrax_gkr::hyrax_output_layer::HyraxOutputLayer;
// use crate::hyrax_gkr::{HyraxCircuit, HyraxProof};

use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;

use rand::RngCore;
use remainder::expression::generic_expr::Expression;
use remainder::expression::prover_expr::ProverExpr;
use remainder::layer::identity_gate::IdentityGate;
use remainder::layer::layer_enum::{CircuitLayerEnum, LayerEnum};
use remainder::layer::matmult::{MatMult, Matrix};
use remainder::layer::regular_layer::RegularLayer;
use remainder::layer::{Layer, LayerId};
use remainder::mle::dense::DenseMle;
use remainder::mle::evals::{Evaluations, MultilinearExtension};
use remainder::mle::mle_enum::MleEnum;
use remainder::mle::Mle;
use remainder_shared_types::transcript::ec_transcript::{
    ECProverTranscript, ECTranscriptReader, ECTranscriptWriter,
};
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    FieldExt, Poseidon,
};
use remainder_shared_types::{transcript::Transcript, Fr};

use super::hyrax_input_layer::HyraxInputLayer;
use super::hyrax_layer::HyraxLayerProof;
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Evaluates (a copy of) the MLE at a given point.
/// Helper function for the tests.
pub fn evaluate_mle<F: FieldExt>(mle: &DenseMle<F>, point: &Vec<F>) -> F {
    let mut mle = mle.clone();
    mle.index_mle_indices(0);
    point.iter().enumerate().for_each(|(i, coord)| {
        mle.fix_variable(i, *coord);
    });
    mle.bookkeeping_table()[0]
}

#[test]
fn test_evaluate_mle() {
    let vals = vec![Fr::from(6), Fr::one(), Fr::from(30), Fr::from(2)];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(vals.clone(), LayerId::Input(0));
    let point = vec![Fr::from(3), Fr::from(5)];
    let eval = (Fr::one() - point[0]) * (Fr::one() - point[1]) * vals[0] +
            point[0] * (Fr::one() - point[1]) * vals[1] + // LSB comes first!
            (Fr::one() - point[0]) * point[1] * vals[2] +
            point[0] * point[1] * vals[3];
    assert_eq!(evaluate_mle(&mle, &point), eval);
}

#[test]
/// This tests a GKR layer with very small values (all small values).
fn degree_one_regular_hyrax_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2; // accounts for beta table
    const NUM_VARS: usize = 2;
    // Construct the expression
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::one(),
            Fr::one() + Fr::one() + Fr::one(),
            Fr::one(),
            Fr::one() + Fr::one(),
        ],
        LayerId::Input(0),
    );
    let expression = mle_1.expression();

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));

    // Other auxiliaries needed for the layer.
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::one(),
            Fr::one() + Fr::one() + Fr::one(),
            Fr::one(),
            Fr::one() + Fr::one(),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(6), Fr::from(5).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(&evaluation_of_mle_at_point, &blinding);

    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Input(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_producing_claim.clone())),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
/// This tests a simple identity gate layer where the MLE that we are "rerouting" has very small values.
/// And has only two variables. The resulting MLE after the rerouting only has one variable.
fn identity_gate_hyrax_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::one(), Fr::from(2_u64), Fr::one(), Fr::from(3_u64)],
        LayerId::Layer(0),
    );
    // The wirings
    let nonzero_gates = vec![(0, 1), (1, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let identity_layer: IdentityGate<Scalar> =
        IdentityGate::new(LayerId::Layer(0), nonzero_gates, mle_1);
    let mut layer_enum = LayerEnum::IdentityGate(Box::new(identity_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS_MLE + 1,
        "hi why is this not working, please help me",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> =
        DenseMle::new_from_raw(vec![Fr::from(2_u64), Fr::from(3)], LayerId::Input(0));
    let claim_point = vec![Fr::from(6)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(&evaluation_of_mle_at_point, &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Layer(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_producing_claim)),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
/// Testing a very simple matmult layer with small values.
/// The two matrices we are multiplying each are 2x2 matrices.
fn matmult_hyrax_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The matrices needed in order to generate the matmult layer.
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::one(), Fr::from(4_u64), Fr::from(3_u64), Fr::from(5_u64)],
        LayerId::Layer(0),
    );
    let matrix_a = Matrix::new(mle_1, 2, 2);
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(7_u64),
            Fr::from(3_u64),
            Fr::from(1_u64),
            Fr::from(2_u64),
        ],
        LayerId::Layer(0),
    );

    // Construct the Matmult layer using the above.
    let matrix_b = Matrix::new(mle_2, 2, 2);
    let matmult_layer: MatMult<Scalar> = MatMult::new(LayerId::Layer(0), matrix_a, matrix_b);
    let mut layer_enum = LayerEnum::MatMult(Box::new(matmult_layer));

    // Other auxiliaries needed for the layer.
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS_MLE + 1,
        "hi why is this not working, please help me",
        None,
    );
    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(11), Fr::from(11), Fr::from(26), Fr::from(19)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(6), Fr::from(5).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(&evaluation_of_mle_at_point, &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Layer(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_producing_claim)),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
/// This test tests a regular layer representing the following expression:
/// Product(`mle_right_1`, `mle_right_2`).
/// Each internal MLE has 2 variables.
fn product_of_mles_regular_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_EXPR: usize = 3;
    // Construct the expression
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(6), Fr::from(1)],
        LayerId::Input(0),
    );
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(3), Fr::from(1), Fr::from(5), Fr::from(2)],
        LayerId::Input(0),
    );
    let expression = Expression::<Fr, ProverExpr>::products(vec![mle_1.clone(), mle_2.clone()]);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));

    // Other auxiliaries needed for the layer.
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS_EXPR + 1,
        "hi why is this not working, please help me",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim = DenseMle::new_from_raw(
        vec![Fr::from(6), Fr::one(), Fr::from(30), Fr::from(2)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(5)];
    let mle_ref = mle_producing_claim;
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval =
        committer.committed_scalar(&evaluate_mle(&mle_ref, &claim_point), &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Input(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_ref.clone())),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
/// This test tests a regular layer representing the following expression:
/// Selector(`mle_left`, `mle_right`).
/// Each internal MLE has 2 variables. Because of the selector variable, the MLE representing
/// the evaluations of the following expression on the boolean hypercube has 3 variables.
fn selector_only_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_EXPR: usize = 3;

    // Construct the expression
    let mle_left: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::one(), Fr::from(3), Fr::one(), Fr::one()],
        LayerId::Input(0),
    );
    let mle_right: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(6), Fr::from(4)],
        LayerId::Input(0),
    );
    let expression_left = mle_left.expression();
    let expression_right = mle_right.expression();
    let selector_expression = expression_right.concat_expr(expression_left);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), selector_expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));

    // Other auxiliaries needed for the layer.
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS_EXPR + 1,
        "hi why is this not working, please help me",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim = DenseMle::new_from_raw(
        vec![
            Fr::one(),
            Fr::from(2),
            Fr::from(3),
            Fr::one(),
            Fr::one(),
            Fr::from(6),
            Fr::one(),
            Fr::from(4),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(5), Fr::from(2), Fr::from(3).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(&evaluation_of_mle_at_point, &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Input(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_producing_claim.clone())),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
/// This test tests a regular layer representing the following expression:
/// Selector(`mle_left`, Product(`mle_right_1`, `mle_right_2`)).
/// Each internal MLE has 2 variables. Because of the selector variable, the MLE representing
/// the evaluations of the following expression on the boolean hypercube has 3 variables.
fn degree_two_selector_regular_hyrax_layer_test() {
    let mut blinding_rng = &mut rand::thread_rng();
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");
    const LAYER_DEGREE: usize = 3;
    const NUM_VARS_EXPR: usize = 3;
    // Construct the expression
    let mle_left: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::one(), Fr::from(3), Fr::one(), Fr::one()],
        LayerId::Input(0),
    );
    let mle_right_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(6), Fr::from(1)],
        LayerId::Input(0),
    );
    let mle_right_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(3), Fr::from(1), Fr::from(5), Fr::from(2)],
        LayerId::Input(0),
    );
    let expression_left = mle_left.expression();
    let expression_right = Expression::<Fr, ProverExpr>::products(vec![mle_right_1, mle_right_2]);
    let selector_expression = expression_right.concat_expr(expression_left);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), selector_expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));

    // Other auxiliaries needed for the layer.
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * NUM_VARS_EXPR + 1,
        "hi why is this not working, please help me",
        None,
    );
    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim = DenseMle::new_from_raw(
        vec![
            Fr::one(),
            Fr::from(6),
            Fr::from(3),
            Fr::one(),
            Fr::one(),
            Fr::from(30),
            Fr::one(),
            Fr::from(2),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(5), Fr::from(2), Fr::from(3).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(&evaluation_of_mle_at_point, &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Input(0),
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(mle_producing_claim.clone())),
        evaluation: commitment_to_eval,
    }];

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the layer to a layer description for the verifier.
    let layer_desc: CircuitLayerEnum<Fr> = layer_enum.into_circuit_layer().unwrap();

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut verifier_transcript,
    );
}

#[test]
fn hyrax_input_layer_proof_test() {
    let mut blinding_rng = &mut rand::thread_rng();
    // type Transcript = PoseidonTranscript<Scalar, Base>;
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test claim agg transcript");

    let claim_point = vec![Fr::from(1983), Fr::from(10832)];
    let committer = PedersenCommitter::<Bn256Point>::new(
        10,
        "hi why is this not working, please help me",
        None,
    );

    let layer_id = LayerId::Input(0);
    let input_mle = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![
            Scalar::from(1093820),
            Scalar::from(21843),
            Scalar::from(47194),
            Scalar::from(1948),
        ],
    ));
    // --- Just for evaluations ---
    let input_dense_mle = DenseMle::new_from_raw(
        vec![
            Scalar::from(1093820),
            Scalar::from(21843),
            Scalar::from(47194),
            Scalar::from(1948),
        ],
        layer_id,
    );

    // --- Create input layer and generate commitment, then add to transcript ---
    let input_layer: HyraxInputLayer<Bn256Point> =
        HyraxInputLayer::new_with_committer(input_mle, layer_id, &committer);
    let hyrax_commitment = input_layer.commit();
    prover_transcript.append_ec_points("Hyrax PCS commit", &hyrax_commitment);

    let commitment_to_eval = committer.committed_scalar(
        &evaluate_mle(&input_dense_mle, &claim_point),
        &input_layer.blinding_factor_eval,
    );

    let claim = HyraxClaim {
        to_layer_id: layer_id,
        point: claim_point,
        mle_enum: Some(MleEnum::Dense(input_dense_mle.clone())),
        evaluation: commitment_to_eval,
    };

    let proof = HyraxInputLayerProof::prove(
        &input_layer,
        &hyrax_commitment,
        &vec![claim.clone()],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    proof.verify(
        &vec![claim.to_claim_commitment()],
        &committer,
        &mut verifier_transcript,
    )
}

// #[test]
// fn small_regular_circuit_hyrax_input_layer_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );
//     let prover_transcript = &mut Transcript::new("");

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer: HyraxInputLayer<Bn256Point, Transcript> = HyraxInputLayer::new_with_committer(
//         input_mle.clone(),
//         LayerId::Input(0),
//         committer.clone(),
//     );
//     let mut input_enum = InputLayerEnum::HyraxInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();
//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);
//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(square_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         &mut Transcript::new(""),
//     )
// }

// #[test]
// fn small_regular_circuit_public_input_layer_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );
//     let prover_transcript = &mut Transcript::new("");

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer =
//         PublicInputLayer::<Bn256Point, _, _>::new(input_mle.clone(), LayerId::Input(0));
//     let mut input_enum = InputLayerEnum::PublicInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit.clone()];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();
//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);
//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(square_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         &mut Transcript::new(""),
//     )
// }

// #[test]
// fn medium_regular_circuit_hyrax_input_layer_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     let prover_transcript = &mut Transcript::new("");
//     let verifier_transcript = &mut Transcript::new("");
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer: HyraxInputLayer<Bn256Point, Transcript> = HyraxInputLayer::new_with_committer(
//         input_mle.clone(),
//         LayerId::Input(0),
//         committer.clone(),
//     );
//     let mut input_enum = InputLayerEnum::HyraxInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();

//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);

//     // Create a layer builder which is sel(square_output + square_output, square_output)
//     let selector_square_builder = from_mle(
//         square_output,
//         // Expression to build
//         |mle| {
//             ExpressionStandard::Mle(mle).concat_expr(
//                 ExpressionStandard::Mle(mle) + ExpressionStandard::Mle(mle),
//             )
//         },
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().flat_map(|eval| vec![eval + eval, *eval]),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let selector_square_output = layers.add_gkr(selector_square_builder);

//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(selector_square_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         verifier_transcript,
//     )
// }

// #[test]
// fn medium_regular_circuit_public_input_layer_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     let prover_transcript = &mut Transcript::new("");
//     let verifier_transcript = &mut Transcript::new("");
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer =
//         PublicInputLayer::<Bn256Point, _, _>::new(input_mle.clone(), LayerId::Input(0));
//     let mut input_enum = InputLayerEnum::PublicInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit.clone()];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();

//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);

//     // Create a layer builder which is sel(square_output + square_output, square_output)
//     let selector_square_builder = from_mle(
//         square_output,
//         // Expression to build
//         |mle| {
//             ExpressionStandard::Mle(mle).concat_expr(
//                 ExpressionStandard::Mle(mle) + ExpressionStandard::Mle(mle),
//             )
//         },
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().flat_map(|eval| vec![eval + eval, *eval]),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let selector_square_output = layers.add_gkr(selector_square_builder);

//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(selector_square_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         verifier_transcript,
//     )
// }

// #[test]
// fn regular_identity_circuit_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     let prover_transcript = &mut Transcript::new("");
//     let verifier_transcript = &mut Transcript::new("");
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer: HyraxInputLayer<Bn256Point, Transcript> = HyraxInputLayer::new_with_committer(
//         input_mle.clone(),
//         LayerId::Input(0),
//         committer.clone(),
//     );
//     let mut input_enum = InputLayerEnum::HyraxInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();
//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);

//     // Create identity gate layer
//     let nonzero_gate_wiring = vec![(0, 2), (1, 1)];
//     let id_output = layers.add_identity_gate(nonzero_gate_wiring, square_output);
//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(id_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         verifier_transcript,
//     )
// }

// #[test]
// fn regular_identity_matmult_circuit_test() {
//     let mut blinding_rng = &mut rand::thread_rng();
//     type Transcript = PoseidonTranscript<Scalar, Base>;
//     let prover_transcript = &mut Transcript::new("");
//     let verifier_transcript = &mut Transcript::new("");
//     const NUM_GENERATORS: usize = 10;
//     let committer = PedersenCommitter::<Bn256Point>::new(
//         NUM_GENERATORS + 1,
//         "hi why is this not working, please help me",
//         None,
//     );

//     // INPUT LAYER CONSTRUCTION
//     let input_mle = DenseMle::<Scalar, Scalar>::new_from_raw(
//         vec![
//             Scalar::from(8797),
//             Scalar::from(7308),
//             Scalar::from(94),
//             Scalar::from(67887),
//         ],
//         LayerId::Input(0),
//         None,
//     );
//     let input_layer: HyraxInputLayer<Bn256Point, Transcript> = HyraxInputLayer::new_with_committer(
//         input_mle.clone(),
//         LayerId::Input(0),
//         committer.clone(),
//     );
//     let mut input_enum = InputLayerEnum::HyraxInputLayer(input_layer);
//     let input_commit = input_enum.commit().unwrap();
//     let input_layers = vec![input_enum];
//     let input_commitments = vec![input_commit];

//     // INTERMEDIATE LAYERS CONSTRUCTION
//     let mut layers: Layers<Scalar, Base, Transcript> = Layers::new();
//     // Create a layer builder which is input_mle * input_mle
//     let squaring_builder = from_mle(
//         input_mle,
//         // Expression to build
//         |mle| ExpressionStandard::products(vec![mle, mle]),
//         // Expected output
//         |mle, layer_id, prefix_bits| {
//             DenseMle::new_from_iter(
//                 mle.mle.iter().map(|eval| eval * eval),
//                 layer_id,
//                 prefix_bits,
//             )
//         },
//     );
//     let square_output = layers.add_gkr(squaring_builder);

//     // Create identity gate layer
//     let nonzero_gate_wiring = vec![(0, 2), (1, 1), (2, 0), (3, 1)];
//     let id_output_for_matrix_a =
//         layers.add_identity_gate(nonzero_gate_wiring, square_output);

//     // Create identity gate layer
//     let nonzero_gate_wiring = vec![(0, 3), (1, 0), (2, 1), (3, 1)];
//     let id_output_for_matrix_b =
//         layers.add_identity_gate(nonzero_gate_wiring, square_output);

//     // Create matmult layer, multiply id_output by itself
//     let matrix_a = Matrix::new(id_output_for_matrix_a, 1, 1);
//     let matrix_b = Matrix::new(id_output_for_matrix_b, 1, 1);
//     let matmult_output = layers.add_matmult_layer(matrix_a, matrix_b);
//     let layers = layers.0;

//     // OUTPUT LAYER CONSTRUCTION
//     let output_layer: HyraxOutputLayer<Bn256Point> = HyraxOutputLayer {
//         underlying_mle: MleEnum::Dense(matmult_output),
//     };
//     let output_layers = vec![output_layer];

//     // FULL CIRCUIT
//     let mut circuit = HyraxCircuit {
//         input_layers,
//         layers,
//         output_layers,
//         input_commitments,
//     };

//     // PROVE
//     let proof = HyraxProof::prove(
//         &mut circuit,
//         &committer,
//         &mut blinding_rng,
//         prover_transcript,
//         &mut VandermondeInverse::new(),
//     );

//     // VERIFY
//     let circuit_description: CircuitDescription<Bn256Point> = circuit.into();
//     HyraxProof::verify(
//         &proof,
//         &circuit_description,
//         &committer,
//         verifier_transcript,
//     )
// }

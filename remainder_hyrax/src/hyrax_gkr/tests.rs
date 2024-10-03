use crate::hyrax_gkr::hyrax_circuit_inputs::HyraxInputLayerData;
use crate::hyrax_gkr::hyrax_input_layer::HyraxInputLayerProof;
use crate::hyrax_gkr::hyrax_layer::HyraxClaim;

use crate::hyrax_gkr::HyraxProver;
use crate::pedersen::{CommittedScalar, PedersenCommitter};
use crate::utils::vandermonde::VandermondeInverse;

use itertools::{repeat_n, Itertools};
use rand::rngs::ThreadRng;
use rand::RngCore;
use remainder::expression::abstract_expr::ExprBuilder;
use remainder::expression::circuit_expr::{ExprDescription, MleDescription};
use remainder::expression::generic_expr::Expression;
use remainder::expression::prover_expr::ProverExpr;
use remainder::layer::identity_gate::{IdentityGate, IdentityGateLayerDescription};
use remainder::layer::layer_enum::{LayerDescriptionEnum, LayerEnum};
use remainder::layer::matmult::{MatMult, MatMultLayerDescription, Matrix, MatrixDescription};
use remainder::layer::regular_layer::{RegularLayer, RegularLayerDescription};
use remainder::layer::{LayerDescription, LayerId};
use remainder::layouter::component::ComponentSet;
use remainder::layouter::nodes::circuit_inputs::{
    InputLayerNode, InputLayerType, InputShred, InputShredData,
};
use remainder::layouter::nodes::circuit_outputs::OutputNode;
use remainder::layouter::nodes::identity_gate::IdentityGateNode;
use remainder::layouter::nodes::matmult::MatMultNode;
use remainder::layouter::nodes::node_enum::NodeEnum;
use remainder::layouter::nodes::sector::Sector;
use remainder::layouter::nodes::{CircuitNode, Context};
use remainder::mle::dense::DenseMle;
use remainder::mle::evals::{Evaluations, MultilinearExtension};
use remainder::mle::{Mle, MleIndex};
use remainder_shared_types::transcript::ec_transcript::{
    ECProverTranscript, ECTranscriptReader, ECTranscriptWriter, ECVerifierTranscript,
};
use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::Fr;
use remainder_shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    Field,
};

use super::hyrax_input_layer::HyraxInputLayer;
use super::hyrax_layer::HyraxLayerProof;
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Evaluates (a copy of) the MLE at a given point.
/// Helper function for the tests.
pub fn evaluate_mle<F: Field>(mle: &DenseMle<F>, point: &Vec<F>) -> F {
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
    let circuit_mle_1 = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, NUM_VARS).collect_vec(),
    );
    let circuit_expr = circuit_mle_1.expression();
    let mut circuit_layer_enum = LayerDescriptionEnum::Regular(RegularLayerDescription::new_raw(
        LayerId::Layer(0),
        circuit_expr,
    ));
    circuit_layer_enum.index_mle_indices(0);

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
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 1), (1, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::IdentityGate(
        IdentityGateLayerDescription::new(LayerId::Layer(0), nonzero_gates.clone(), circuit_mle_1),
    );
    circuit_layer_enum.index_mle_indices(0);
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
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let matrix_a = Matrix::new(mle_1, 1, 1);
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(7_u64),
            Fr::from(3_u64),
            Fr::from(1_u64),
            Fr::from(2_u64),
        ],
        LayerId::Layer(0),
    );
    let matrix_b = Matrix::new(mle_2, 1, 1);
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );
    let circuit_matrix_a = MatrixDescription::new(circuit_mle_1, 1, 1);
    let circuit_matrix_b = MatrixDescription::new(circuit_mle_2, 1, 1);
    let mut circuit_layer_enum = LayerDescriptionEnum::MatMult(MatMultLayerDescription::new(
        LayerId::Layer(0),
        circuit_matrix_a,
        circuit_matrix_b,
    ));
    circuit_layer_enum.index_mle_indices(0);

    // Construct the Matmult layer using the above.

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
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_expr =
        Expression::<Fr, ExprDescription>::products(vec![circuit_mle_1, circuit_mle_2]);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));
    let mut circuit_layer_enum = LayerDescriptionEnum::Regular(RegularLayerDescription::new_raw(
        LayerId::Layer(0),
        circuit_expr,
    ));
    circuit_layer_enum.index_mle_indices(0);

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
    let mle = mle_producing_claim;
    let blinding = Fr::from(blinding_rng.next_u64());
    let commitment_to_eval =
        committer.committed_scalar(&evaluate_mle(&mle, &claim_point), &blinding);
    let claims: Vec<HyraxClaim<Fr, CommittedScalar<Bn256Point>>> = vec![HyraxClaim {
        to_layer_id: LayerId::Input(0),
        point: claim_point,
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let circuit_mle_left = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_mle_right = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_expression_left = circuit_mle_left.expression();
    let circuit_expression_right = circuit_mle_right.expression();
    let selector_circuit_expression = circuit_expression_left.select(circuit_expression_right);
    let expression_left = mle_left.expression();
    let expression_right = mle_right.expression();
    let selector_expression = expression_left.select(expression_right);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), selector_expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));
    let mut circuit_layer_enum = LayerDescriptionEnum::Regular(RegularLayerDescription::new_raw(
        LayerId::Layer(0),
        selector_circuit_expression,
    ));
    circuit_layer_enum.index_mle_indices(0);

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
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let circuit_mle_left = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_mle_right_1 = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let circuit_mle_right_2 = MleDescription::new(
        LayerId::Input(0),
        &repeat_n(MleIndex::Free, 2).collect_vec(),
    );
    let expression_left = mle_left.expression();
    let circuit_expression_left = circuit_mle_left.expression();
    let expression_right = Expression::<Fr, ProverExpr>::products(vec![mle_right_1, mle_right_2]);
    let circuit_expression_right =
        Expression::<Fr, ExprDescription>::products(vec![circuit_mle_right_1, circuit_mle_right_2]);
    let selector_expression = expression_left.select(expression_right);
    let circuit_selector_expression = circuit_expression_left.select(circuit_expression_right);

    // Construct the GKR Layer from the expression
    let layer: RegularLayer<Scalar> = RegularLayer::new_raw(LayerId::Layer(0), selector_expression);
    let mut layer_enum = LayerEnum::Regular(Box::new(layer));
    let mut circuit_layer_enum = LayerDescriptionEnum::Regular(RegularLayerDescription::new_raw(
        LayerId::Layer(0),
        circuit_selector_expression,
    ));
    circuit_layer_enum.index_mle_indices(0);

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
        evaluation: commitment_to_eval,
    }];

    // Convert the layer to a layer description for the verifier.
    let layer_desc: LayerDescriptionEnum<Fr> = circuit_layer_enum;

    // Construct the layer proof
    let (hyrax_layer_proof, _) = HyraxLayerProof::prove(
        &mut layer_enum,
        &claims,
        &[mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

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
    let mut input_layer: HyraxInputLayer<Bn256Point> =
        HyraxInputLayer::new_with_committer(input_mle, layer_id, &committer, &None);
    let hyrax_commitment = input_layer.commit();
    prover_transcript.append_ec_points("Hyrax PCS commit", &hyrax_commitment);

    let commitment_to_eval = committer.committed_scalar(
        &evaluate_mle(&input_dense_mle, &claim_point),
        &input_layer.blinding_factor_eval,
    );

    let claim = HyraxClaim {
        to_layer_id: layer_id,
        point: claim_point,
        evaluation: commitment_to_eval,
    };

    let proof = HyraxInputLayerProof::prove(
        &input_layer,
        &hyrax_commitment,
        &[claim.clone()],
        &committer,
        &mut blinding_rng,
        &mut prover_transcript,
        &mut VandermondeInverse::new(),
    );

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    // Consume the commitment from the verifier transcript.
    let _hyrax_commitment: Vec<Bn256Point> = verifier_transcript
        .consume_ec_points("Hyrax PCS commit", hyrax_commitment.len())
        .unwrap();
    proof.verify(
        &[claim.to_claim_commitment()],
        &committer,
        &mut verifier_transcript,
    )
}

#[test]
fn small_regular_circuit_hyrax_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::HyraxInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Middle layer 2: subtract middle layer 1 from itself.
        let subtract_sector = Sector::new(ctx, &[&&squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn small_regular_circuit_public_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Middle layer 2: subtract middle layer 1 from itself.
        let subtract_sector = Sector::new(ctx, &[&&squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn medium_regular_circuit_hyrax_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::HyraxInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Middle layer 2: Create a layer builder which is sel(square_output + square_output, square_output)
        let selector_squaring_sector = Sector::new(ctx, &[&&squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            (mle.expr() + mle.expr()).select(mle.expr())
        });

        // Middle layer 3: subtract middle layer 2 from itself.
        let subtract_sector = Sector::new(ctx, &[&&selector_squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                selector_squaring_sector.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn medium_regular_circuit_public_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Middle layer 2: Create a layer builder which is sel(square_output * square_output, square_output)
        let selector_squaring_sector = Sector::new(ctx, &[&&squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle]).select(mle.expr())
        });

        // Middle layer 3: subtract middle layer 2 from itself.
        let subtract_sector = Sector::new(ctx, &[&&selector_squaring_sector], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                selector_squaring_sector.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn matmult_hyrax_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::HyraxInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        let matmult_layer = MatMultNode::new(ctx, &input_shred, (1, 1), &input_shred, (1, 1));

        // Middle layer 1: subtract middle layer 0 from itself.
        let subtract_sector = Sector::new(ctx, &[&matmult_layer], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                matmult_layer.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn regular_identity_hyrax_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::HyraxInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Create identity gate layer
        let nonzero_gate_wiring = vec![(0, 2), (1, 1)];
        let id_layer = IdentityGateNode::new(ctx, &squaring_sector, nonzero_gate_wiring);

        // Middle layer 2: subtract middle layer 1 from itself.
        let subtract_sector = Sector::new(ctx, &[&id_layer], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                id_layer.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

#[test]
fn regular_identity_matmult_hyrax_input_layer_test() {
    let mut prover_transcript: ECTranscriptWriter<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptWriter::new("Test small regular circuit");
    let blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    let mut hyrax_prover: HyraxProver<Bn256Point, _, ThreadRng> =
        HyraxProver::new(&committer, blinding_rng, converter);

    let mut circuit_function: &mut dyn FnMut(
        &Context,
    ) -> (
        ComponentSet<NodeEnum<Fr>>,
        Vec<HyraxInputLayerData<Bn256Point>>,
    ) = &mut |ctx| {
        // INPUT LAYER CONSTRUCTION
        let input_multilinear_extension = MultilinearExtension::new(vec![
            Scalar::from(8797),
            Scalar::from(7308),
            Scalar::from(94),
            Scalar::from(67887),
        ]);
        let input_layer = InputLayerNode::new(ctx, None, InputLayerType::HyraxInputLayer);
        let input_shred =
            InputShred::new(ctx, input_multilinear_extension.num_vars(), &input_layer);
        let input_shred_data = InputShredData::new(input_shred.id(), input_multilinear_extension);
        let input_layer_data =
            HyraxInputLayerData::new(input_layer.id(), vec![input_shred_data], None, None);

        // Middle layer 1: square the input.
        let squaring_sector = Sector::new(ctx, &[&input_shred], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            ExprBuilder::products(vec![mle, mle])
        });

        // Create identity gate layer A
        let nonzero_gate_wiring_a = vec![(0, 2), (1, 1), (2, 0), (3, 1)];
        let id_layer_a = IdentityGateNode::new(ctx, &squaring_sector, nonzero_gate_wiring_a);

        // Create identity gate layer B
        let nonzero_gate_wiring_b = vec![(0, 3), (1, 0), (2, 1), (3, 1)];
        let id_layer_b = IdentityGateNode::new(ctx, &squaring_sector, nonzero_gate_wiring_b);

        // Create matmult layer, multiply id_output by itself
        let matmult_layer = MatMultNode::new(ctx, &id_layer_a, (1, 1), &id_layer_b, (1, 1));

        // Middle layer 5: subtract middle layer 4 from itself.
        let subtract_sector = Sector::new(ctx, &[&matmult_layer], |mle_vec| {
            assert_eq!(mle_vec.len(), 1);
            let mle = mle_vec[0];
            mle.expr() - mle.expr()
        });

        // Make this an output node.
        let output_node = OutputNode::new_zero(ctx, &subtract_sector);

        (
            ComponentSet::<NodeEnum<Fr>>::new_raw(vec![
                input_layer.into(),
                input_shred.into(),
                squaring_sector.into(),
                id_layer_a.into(),
                id_layer_b.into(),
                matmult_layer.into(),
                subtract_sector.into(),
                output_node.into(),
            ]),
            vec![input_layer_data],
        )
    };

    let (mut circuit_description, proof) =
        hyrax_prover.prove_gkr_circuit(&mut circuit_function, &mut prover_transcript);

    let mut verifier_transcript: ECTranscriptReader<Bn256Point, PoseidonSponge<Base>> =
        ECTranscriptReader::new(prover_transcript.get_transcript());

    hyrax_prover.verify_gkr_circuit(&proof, &mut circuit_description, &mut verifier_transcript);
}

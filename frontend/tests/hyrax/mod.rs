use ark_std::test_rng;
use frontend::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};
use hyrax::gkr::input_layer::{
    commit_to_input_values, HyraxInputLayerDescription, HyraxInputLayerProof,
};
use hyrax::gkr::layer::{HyraxClaim, HyraxLayerProof};
use hyrax::gkr::verify_hyrax_proof;
use hyrax::utils::vandermonde::VandermondeInverse;
use itertools::{repeat_n, Itertools};
use rand::RngCore;
use remainder::expression::circuit_expr::ExprDescription;
use remainder::expression::generic_expr::Expression;
use remainder::expression::prover_expr::ProverExpr;
use remainder::layer::gate::{BinaryOperation, GateLayer, GateLayerDescription};
use remainder::layer::identity_gate::{IdentityGate, IdentityGateLayerDescription};
use remainder::layer::layer_enum::{LayerDescriptionEnum, LayerEnum};
use remainder::layer::matmult::{MatMult, MatMultLayerDescription, Matrix, MatrixDescription};
use remainder::layer::regular_layer::{RegularLayer, RegularLayerDescription};
use remainder::layer::{LayerDescription, LayerId};
use remainder::mle::dense::DenseMle;
use remainder::mle::evals::{Evaluations, MultilinearExtension};
use remainder::mle::mle_description::MleDescription;
use remainder::mle::{Mle, MleIndex};
use remainder::utils::mle::get_random_mle;
use shared_types::config::{GKRCircuitProverConfig, GKRCircuitVerifierConfig};
use shared_types::transcript::ec_transcript::{ECTranscript, ECTranscriptTrait};
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::{
    ff_field, perform_function_under_expected_configs, perform_function_under_prover_config,
};
use shared_types::{
    halo2curves::{bn256::G1 as Bn256Point, group::Group, CurveExt},
    pedersen::{CommittedScalar, PedersenCommitter},
    Field,
};
use shared_types::{perform_function_under_verifier_config, Fr};
type Scalar = <Bn256Point as Group>::Scalar;
type Base = <Bn256Point as CurveExt>::Base;

/// Evaluates (a copy of) the MLE at a given point.
/// Helper function for the tests.
pub fn evaluate_mle<F: Field>(mle: &DenseMle<F>, point: &[F]) -> F {
    let mut mle = mle.clone();
    mle.index_mle_indices(0);
    point.iter().enumerate().for_each(|(i, coord)| {
        mle.fix_variable(i, *coord);
    });
    mle.value()
}

#[test]
fn test_evaluate_mle() {
    let vals = vec![Fr::from(6), Fr::one(), Fr::from(30), Fr::from(2)];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(vals.clone(), LayerId::Input(0));
    let point = vec![Fr::from(3), Fr::from(5)];
    let eval = (Fr::one() - point[0]) * (Fr::one() - point[1]) * vals[0]
        + point[0] * (Fr::one() - point[1]) * vals[2]
        + (Fr::one() - point[0]) * point[1] * vals[1]
        + point[0] * point[1] * vals[3];
    assert_eq!(evaluate_mle(&mle, &point), eval);
}

#[test]
fn degree_one_regular_hyrax_layer_test() {
    let prover_config = GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let verifier_config = GKRCircuitVerifierConfig::new_from_prover_config(&prover_config, false);
    perform_function_under_expected_configs!(
        helper_degree_one_regular_hyrax_layer_test,
        &prover_config,
        &verifier_config,
    )
}
/// This tests a GKR layer with very small values (all small values).
fn helper_degree_one_regular_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
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
    let circuit_expr = Expression::from_mle_desc(circuit_mle_1);
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
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This tests a simple identity gate layer where the MLE that we are "rerouting" has very small values.
/// And has only two variables. The resulting MLE after the rerouting only has one variable.
fn identity_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

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
    let mut circuit_layer_enum =
        LayerDescriptionEnum::IdentityGate(IdentityGateLayerDescription::new(
            LayerId::Layer(0),
            nonzero_gates.clone(),
            circuit_mle_1,
            1,
            None,
        ));
    circuit_layer_enum.index_mle_indices(0);
    let identity_layer: IdentityGate<Scalar> =
        IdentityGate::new(LayerId::Layer(0), nonzero_gates, mle_1, 1, 0);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is a dataparallel version of the [`identity_gate_hyrax_layer_test`]
/// The input MLE has four (two dataparallel) variables. The resulting MLE after the
/// rerouting only has three (two dataparallel) variables.
fn dataparallel_uneven_identity_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();

    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 1;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(5),
            Fr::from(2),
            Fr::from(6),
            Fr::from(3),
            Fr::from(7),
            Fr::from(4),
            Fr::from(8),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 1)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum =
        LayerDescriptionEnum::IdentityGate(IdentityGateLayerDescription::new(
            LayerId::Layer(0),
            nonzero_gates.clone(),
            circuit_mle_1,
            DATAPARALLEL_NUM_VARS_MLE,
            Some(DATAPARALLEL_NUM_VARS_MLE),
        ));
    circuit_layer_enum.index_mle_indices(0);
    let identity_layer: IdentityGate<Scalar> = IdentityGate::new(
        LayerId::Layer(0),
        nonzero_gates,
        mle_1,
        DATAPARALLEL_NUM_VARS_MLE,
        DATAPARALLEL_NUM_VARS_MLE,
    );
    let mut layer_enum = LayerEnum::IdentityGate(Box::new(identity_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> =
        DenseMle::new_from_raw(vec![Fr::from(5), Fr::from(7)], LayerId::Input(0));
    let claim_point = vec![Fr::one() + Fr::one()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is an "even" version of the [`identity_gate_hyrax_layer_test`]
/// Meaning the input MLE has four (two dataparallel) variables. And the resulting MLE
/// after the rerouting also has four (two dataparallel) variables.
fn dataparallel_even_identity_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(5),
            Fr::from(2),
            Fr::from(6),
            Fr::from(3),
            Fr::from(7),
            Fr::from(4),
            Fr::from(8),
            Fr::from(6),
            Fr::from(3),
            Fr::from(7),
            Fr::from(7),
            Fr::from(2),
            Fr::from(6),
            Fr::from(3),
            Fr::from(7),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 3), (1, 2), (2, 1), (3, 0)];
    // let nonzero_gates = vec![(0, 1), (1, 0)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum =
        LayerDescriptionEnum::IdentityGate(IdentityGateLayerDescription::new(
            LayerId::Layer(0),
            nonzero_gates.clone(),
            circuit_mle_1,
            NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE,
            Some(DATAPARALLEL_NUM_VARS_MLE),
        ));
    circuit_layer_enum.index_mle_indices(0);
    let identity_layer: IdentityGate<Scalar> = IdentityGate::new(
        LayerId::Layer(0),
        nonzero_gates,
        mle_1,
        NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE,
        DATAPARALLEL_NUM_VARS_MLE,
    );
    let mut layer_enum = LayerEnum::IdentityGate(Box::new(identity_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(6),
            Fr::from(2),
            Fr::from(5),
            Fr::from(1),
            Fr::from(8),
            Fr::from(4),
            Fr::from(7),
            Fr::from(3),
            Fr::from(7),
            Fr::from(7),
            Fr::from(3),
            Fr::from(6),
            Fr::from(7),
            Fr::from(3),
            Fr::from(6),
            Fr::from(2),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(2), Fr::from(5), Fr::from(7), Fr::from(3)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// Adds the lhs MLE and rhs MLE and reroutes the result to the output MLE.
/// Both input MLEs have two variables. And the resulting MLE
/// after the rerouting also has two variables. Its bookkeeping table is the
/// element-wise sum of the input MLEs
fn even_add_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(4), Fr::from(2), Fr::from(5)],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        None,
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Add,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        None,
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Add,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * NUM_VARS_MLE + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(6), Fr::from(5), Fr::from(9)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// Multiplies the lhs MLE and rhs MLE and reroutes the result to the output MLE.
/// Both input MLEs have two variables. And the resulting MLE
/// after the rerouting also has two variables. Its bookkeeping table is the
/// element-wise sum of the input MLEs
fn even_mul_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(4), Fr::from(2), Fr::from(5)],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        None,
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Mul,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        None,
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Mul,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * NUM_VARS_MLE + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(8), Fr::from(6), Fr::from(20)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(2), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// Adds the first halves of lhs MLE and rhs MLE and reroutes the result to
/// the output MLE. Both input MLEs have two variables. And the resulting MLE
/// after the rerouting also has one variable. Its bookkeeping table is the
/// element-wise sum of the input MLEs' firt halves.
fn uneven_add_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(3), Fr::from(2), Fr::from(5), Fr::from(1)],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(4), Fr::from(5)],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        None,
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Add,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        None,
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Add,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * NUM_VARS_MLE + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> =
        DenseMle::new_from_raw(vec![Fr::from(5), Fr::from(3)], LayerId::Input(0));
    let claim_point = vec![Fr::from(6)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// Multiplies the first halves of lhs MLE and rhs MLE and reroutes the result to
/// the output MLE. Both input MLEs have two variables. And the resulting MLE
/// after the rerouting also has one variable. Its bookkeeping table is the
/// element-wise sum of the input MLEs' firt halves.
fn uneven_mul_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(3), Fr::from(2), Fr::from(5), Fr::from(1)],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(4), Fr::from(5)],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        None,
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Mul,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        None,
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Mul,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * NUM_VARS_MLE + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> =
        DenseMle::new_from_raw(vec![Fr::from(6), Fr::from(2)], LayerId::Input(0));
    let claim_point = vec![Fr::from(6)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is a dataparallel version of the [`even_add_gate_hyrax_layer_test`]
/// Meaning the input MLE has four (two dataparallel) variables. And the resulting MLE
/// after the rerouting also has four (two dataparallel) variables.
fn dataparallel_even_add_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 2;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 1;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(4),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(2),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Add,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Add,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(2),
            Fr::from(6),
            Fr::from(5),
            Fr::from(9),
            Fr::from(5),
            Fr::from(3),
            Fr::from(7),
            Fr::from(7),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(2), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is a dataparallel version of the [`even_mul_gate_hyrax_layer_test`]
/// Meaning the input MLE has four (two dataparallel) variables. And the resulting MLE
/// after the rerouting also has four (two dataparallel) variables.
fn dataparallel_even_mul_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 3;
    const NUM_VARS_MLE: usize = 2;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 1;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(4),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(2),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Mul,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Mul,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(8),
            Fr::from(6),
            Fr::from(20),
            Fr::from(6),
            Fr::from(2),
            Fr::from(6),
            Fr::from(12),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(2), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is a dataparallel version of the [`uneven_add_gate_hyrax_layer_test`]
/// Meaning the input MLE has four (two dataparallel) variables. And the resulting MLE
/// after the rerouting also has four (two dataparallel) variables.
fn dataparallel_uneven_add_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 2;
    const NUM_VARS_MLE: usize = 1;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(4),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(2),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Add,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Add,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(5), Fr::from(5), Fr::from(7)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This is a dataparallel version of the [`uneven_mul_gate_hyrax_layer_test`]
/// Meaning the input MLE has four (two dataparallel) variables. And the resulting MLE
/// after the rerouting also has four (two dataparallel) variables.
fn dataparallel_uneven_mul_gate_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let mut blinding_rng = &mut rand::thread_rng();
    const LAYER_DEGREE: usize = 3;
    const NUM_VARS_MLE: usize = 1;
    const DATAPARALLEL_NUM_VARS_MLE: usize = 2;

    // The MLE we are going to reroute
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(4),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_1 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The MLE we are going to reroute
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(2),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
        ],
        LayerId::Layer(0),
    );
    let circuit_mle_2 = MleDescription::new(
        LayerId::Layer(0),
        &repeat_n(MleIndex::Free, NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE).collect_vec(),
    );

    // The wirings
    let nonzero_gates = vec![(0, 0, 0)];

    // Construct the layer from the underlying MLE and the wirings
    let mut circuit_layer_enum = LayerDescriptionEnum::Gate(GateLayerDescription::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates.clone(),
        circuit_mle_1,
        circuit_mle_2,
        LayerId::Layer(0),
        BinaryOperation::Mul,
    ));
    circuit_layer_enum.index_mle_indices(0);
    let gate_layer: GateLayer<Scalar> = GateLayer::new(
        Some(DATAPARALLEL_NUM_VARS_MLE),
        nonzero_gates,
        mle_1,
        mle_2,
        BinaryOperation::Mul,
        LayerId::Layer(0),
    );
    let mut layer_enum = LayerEnum::Gate(Box::new(gate_layer));

    // Other auxiliaries for the layer
    let committer = PedersenCommitter::<Bn256Point>::new(
        (LAYER_DEGREE + 1) * 2 * (NUM_VARS_MLE + DATAPARALLEL_NUM_VARS_MLE) + 1,
        "not working??not working??not working??not working??",
        None,
    );

    // The MLE representing the expression above evaluated at the boolean hypercube.
    let mle_producing_claim: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(6), Fr::from(6), Fr::from(6)],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(3), Fr::from(2)];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// Testing a very simple matmult layer with small values.
/// The two matrices we are multiplying each are 2x2 matrices.
fn matmult_hyrax_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
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
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This test tests a regular layer representing the following expression:
/// Product(`mle_right_1`, `mle_right_2`).
/// Each internal MLE has 2 variables.
fn product_of_mles_regular_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
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
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
/// This test tests a regular layer representing the following expression:
/// Selector(`mle_left`, `mle_right`).
/// Each internal MLE has 2 variables. Because of the selector variable, the MLE representing
/// the evaluations of the following expression on the boolean hypercube has 3 variables.
fn selector_only_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
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
    let circuit_expression_left = Expression::from_mle_desc(circuit_mle_left);
    let circuit_expression_right = Expression::from_mle_desc(circuit_mle_right);
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
            Fr::from(3),
            Fr::one(),
            Fr::one(),
            Fr::from(2),
            Fr::one(),
            Fr::from(6),
            Fr::from(4),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(5), Fr::from(2), Fr::from(3).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
fn degree_two_selector_regular_hyrax_layer_test() {
    let prover_config = GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let verifier_config = GKRCircuitVerifierConfig::new_from_prover_config(&prover_config, false);
    perform_function_under_expected_configs!(
        helper_degree_two_selector_regular_hyrax_layer_test,
        &prover_config,
        &verifier_config,
    )
}
/// This test tests a regular layer representing the following expression:
/// Selector(`mle_left`, Product(`mle_right_1`, `mle_right_2`)).
/// Each internal MLE has 2 variables. Because of the selector variable, the MLE representing
/// the evaluations of the following expression on the boolean hypercube has 3 variables.
fn helper_degree_two_selector_regular_hyrax_layer_test() {
    let mut blinding_rng = &mut rand::thread_rng();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
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
    let circuit_expression_left = Expression::from_mle_desc(circuit_mle_left);
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
            Fr::from(3),
            Fr::one(),
            Fr::one(),
            Fr::from(6),
            Fr::one(),
            Fr::from(30),
            Fr::from(2),
        ],
        LayerId::Input(0),
    );
    let claim_point = vec![Fr::from(5), Fr::from(2), Fr::from(3).neg()];
    let evaluation_of_mle_at_point = evaluate_mle(&mle_producing_claim, &claim_point);
    let blinding = Fr::random(&mut blinding_rng);
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
        vec![mle_producing_claim],
        &committer,
        &mut blinding_rng,
        &mut transcript,
        &mut VandermondeInverse::new(),
    );

    // Convert the claims into their respective commitments for the verifier view.
    let claim_commitments: Vec<_> = claims
        .iter()
        .map(|claim| claim.to_claim_commitment())
        .collect();

    // Verify
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    HyraxLayerProof::verify(
        &hyrax_layer_proof,
        &layer_desc,
        &claim_commitments,
        &committer,
        &mut transcript,
    );
}

#[test]
fn hyrax_input_layer_proof_test() {
    let mut blinding_rng = &mut rand::thread_rng();
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

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
            Scalar::from(47194),
            Scalar::from(21843),
            Scalar::from(1948),
        ],
    ));
    // Just for evaluations
    let input_dense_mle = DenseMle::new_from_raw(
        vec![
            Scalar::from(1093820),
            Scalar::from(47194),
            Scalar::from(21843),
            Scalar::from(1948),
        ],
        layer_id,
    );

    let input_layer_desc = HyraxInputLayerDescription::new(layer_id, input_mle.num_vars());
    let mut prover_commitment = commit_to_input_values(
        input_layer_desc.num_vars,
        input_layer_desc.log_num_cols,
        &input_mle,
        &committer,
        blinding_rng,
    );

    transcript.append_ec_points("Hyrax PCS commit", &prover_commitment.commitment);

    let blinding_factor_eval = Scalar::from(blinding_rng.next_u64());
    let commitment_to_eval = committer.committed_scalar(
        &evaluate_mle(&input_dense_mle, &claim_point),
        &blinding_factor_eval,
    );

    let claim = HyraxClaim {
        to_layer_id: layer_id,
        point: claim_point,
        evaluation: commitment_to_eval,
    };

    let proof = HyraxInputLayerProof::prove(
        &input_layer_desc,
        &mut prover_commitment,
        &[claim.clone()],
        &committer,
        &mut blinding_rng,
        &mut transcript,
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    // Consume the commitment from the verifier transcript.
    let verifier_commitment = prover_commitment.commitment;
    transcript.append_ec_points("Hyrax PCS commit", &verifier_commitment);
    proof.verify(
        &input_layer_desc,
        &[claim.to_claim_commitment()],
        &committer,
        &mut transcript,
    )
}

#[test]
fn small_regular_circuit_hyrax_input_layer_test() {
    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );
    // INPUT LAYER CONSTRUCTION
    let input_multilinear_extension: MultilinearExtension<Scalar> =
        vec![8797, 7308, 94, 67887].into();

    let mut builder = CircuitBuilder::<Fr>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Committed);
    let input_shred = builder.add_input_shred(
        "Input",
        input_multilinear_extension.num_vars(),
        &input_layer,
    );

    // Middle layer 1: square the input.
    let squaring_sector = builder.add_sector(input_shred.expr() * input_shred.expr());

    // Middle layer 2: subtract middle layer 1 from itself.
    let subtract_sector = builder.add_sector(squaring_sector.expr() - squaring_sector.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    // circuit_desc, input_layer_id_to_input_shred_ids, circuit_description_map
    let mut small_circuit = builder.build_with_layer_combination().unwrap();
    small_circuit.set_input("Input", input_multilinear_extension);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // Compute actual Hyrax proof
    let mut hyrax_provable_circuit = small_circuit.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // For testing only
    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut transcript
    );

    let mut transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut transcript,
        &proof_config
    );
}

/// Creates the [Circuit<F>] for ease of re-use.
fn build_small_regular_test_circuit<F: Field>(num_free_vars: usize) -> Circuit<F> {
    let mut builder = CircuitBuilder::new();

    // All inputs are public inputs
    let public_input_layer_node = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    // Circuit inputs
    let input_mle_shred =
        builder.add_input_shred("Input MLE", num_free_vars, &public_input_layer_node);

    // Create the circuit components
    // Middle layer 1: square the input.
    let squaring_sector = builder.add_sector(input_mle_shred.expr() * input_mle_shred.expr());

    // Middle layer 2: subtract middle layer 1 from itself.
    let subtract_sector = builder.add_sector(squaring_sector.expr() - squaring_sector.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    builder.build_with_layer_combination().unwrap()
}

#[test]
fn small_regular_circuit_public_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_FREE_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_FREE_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc = build_small_regular_test_circuit(NUM_FREE_VARS);
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function for ease of proving.
///
/// Note that this function also returns the [LayerId] of its input layer
/// to be used later for committed input layer specification if needed.
fn build_medium_regular_test_circuit<F: Field>(
    num_free_vars: usize,
    input_layer_vis: LayerVisibility,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // There is only one input layer; it can be public or committed (for the purposes of testing)
    let input_layer_node = builder.add_input_layer("Input Layer", input_layer_vis);

    // Circuit inputs
    let input_mle_shred = builder.add_input_shred("Input MLE", num_free_vars, &input_layer_node);

    // Create the circuit components
    // Middle layer 1: square the input.
    let squaring_sector = builder.add_sector(input_mle_shred.expr() * input_mle_shred.expr());

    // Middle layer 2: Create a layer builder which is sel(square_output + square_output, square_output)
    let selector_squaring_sector = builder.add_sector(
        (squaring_sector.expr() + squaring_sector.expr()).select(squaring_sector.expr()),
    );

    // Middle layer 3: subtract middle layer 2 from itself.
    let subtract_sector =
        builder.add_sector(selector_squaring_sector.expr() - selector_squaring_sector.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    builder.build().unwrap()
}

#[test]
fn medium_regular_circuit_hyrax_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_FREE_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_FREE_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc =
        build_medium_regular_test_circuit(NUM_FREE_VARS, LayerVisibility::Committed);
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

#[test]
fn medium_regular_circuit_public_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_FREE_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_FREE_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc =
        build_medium_regular_test_circuit(NUM_FREE_VARS, LayerVisibility::Public);
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function for ease of proving.
fn build_identity_regular_test_circuit<F: Field>(
    num_free_vars: usize,
    input_layer_vis: LayerVisibility,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // There is only one input layer; it can be public or committed (for the purposes of testing)
    let input_layer_node = builder.add_input_layer("Input Layer", input_layer_vis);

    // Circuit inputs
    let input_mle_shred = builder.add_input_shred("Input MLE", num_free_vars, &input_layer_node);

    // Create the circuit components
    // Middle layer 1: square the input.
    let squaring_sector = builder.add_sector(input_mle_shred.expr() * input_mle_shred.expr());

    // Create identity gate layer
    let nonzero_gate_wiring = vec![(0, 2), (1, 1)];
    let id_layer = builder.add_identity_gate_node(&squaring_sector, nonzero_gate_wiring, 1, None);

    // Middle layer 2: subtract middle layer 1 from itself.
    let subtract_sector = builder.add_sector(id_layer.expr() - id_layer.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    builder.build().unwrap()
}

#[test]
fn identity_public_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 10;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_FREE_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_FREE_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc =
        build_identity_regular_test_circuit(NUM_FREE_VARS, LayerVisibility::Committed);
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function for ease of proving.
fn build_matmult_regular_test_circuit<F: Field>(
    num_row_vars: usize,
    num_col_vars: usize,
    input_layer_vis: LayerVisibility,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // There is only one input layer; it can be public or committed (for the purposes of testing)
    let input_layer_node = builder.add_input_layer("Input Layer", input_layer_vis);

    // Circuit inputs
    let input_mle_shred =
        builder.add_input_shred("Input MLE", num_row_vars + num_col_vars, &input_layer_node);

    // Create the circuit components
    let matmult_layer = builder.add_matmult_node(
        &input_mle_shred,
        (num_row_vars, num_col_vars),
        &input_mle_shred,
        (num_row_vars, num_col_vars),
    );

    // Middle layer 1: subtract middle layer 0 from itself.
    let subtract_sector = builder.add_sector(matmult_layer.expr() - matmult_layer.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    builder.build().unwrap()
}

#[test]
fn regular_matmult_hyrax_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 12;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_ROW_VARS: usize = 2;
    const NUM_COL_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_ROW_VARS + NUM_COL_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc =
        build_matmult_regular_test_circuit(NUM_ROW_VARS, NUM_COL_VARS, LayerVisibility::Committed);
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

/// Creates the [GKRCircuitDescription] and an associated helper input
/// function for ease of proving.
fn build_identity_matmult_regular_test_circuit<F: Field>(
    num_row_vars: usize,
    num_col_vars: usize,
    input_layer_vis: LayerVisibility,
) -> Circuit<F> {
    let mut builder = CircuitBuilder::<F>::new();

    // There is only one input layer; it can be public or committed (for the purposes of testing)
    let input_layer_node = builder.add_input_layer("Input Layer", input_layer_vis);

    // Circuit inputs
    let input_mle_shred =
        builder.add_input_shred("Input MLE", num_row_vars + num_col_vars, &input_layer_node);

    // Create the circuit components
    // Middle layer 1: square the input.
    let squaring_sector = builder.add_sector(input_mle_shred.expr() * input_mle_shred.expr());

    // Create identity gate layer A
    let nonzero_gate_wiring_a = vec![(0, 2), (1, 1), (2, 0), (3, 1)];
    let id_layer_a =
        builder.add_identity_gate_node(&squaring_sector, nonzero_gate_wiring_a, 2, None);

    // Create identity gate layer B
    let nonzero_gate_wiring_b = vec![(0, 3), (1, 0), (2, 1), (3, 1)];
    let id_layer_b =
        builder.add_identity_gate_node(&squaring_sector, nonzero_gate_wiring_b, 2, None);

    // Create matmult layer, multiply id_output by itself
    let matmult_layer = builder.add_matmult_node(&id_layer_a, (1, 1), &id_layer_b, (1, 1));

    // Middle layer 5: subtract middle layer 4 from itself.
    let subtract_sector = builder.add_sector(matmult_layer.expr() - matmult_layer.expr());

    // Make this an output node.
    let _output_node = builder.set_output(&subtract_sector);

    builder.build().unwrap()
}

#[test]
fn regular_identity_matmult_public_input_layer_test() {
    let mut blinding_rng = rand::thread_rng();
    let converter: &mut VandermondeInverse<Scalar> = &mut VandermondeInverse::new();
    const NUM_GENERATORS: usize = 30;
    let committer = PedersenCommitter::<Bn256Point>::new(
        NUM_GENERATORS + 1,
        "hi why is this not working, please help me",
        None,
    );

    // Generate input data
    const NUM_ROW_VARS: usize = 2;
    const NUM_COL_VARS: usize = 2;
    let mut rng = test_rng();
    let input_mle = get_random_mle(NUM_ROW_VARS + NUM_COL_VARS, &mut rng).mle;

    // Create circuit description + semantic input mapping
    let mut circuit_desc = build_identity_matmult_regular_test_circuit(
        NUM_ROW_VARS,
        NUM_COL_VARS,
        LayerVisibility::Committed,
    );
    circuit_desc.set_input("Input MLE", input_mle);

    // --- Create GKR circuit prover + verifier configs which work with Hyrax ---
    let gkr_circuit_prover_config =
        GKRCircuitProverConfig::hyrax_compatible_runtime_optimized_default();
    let gkr_circuit_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&gkr_circuit_prover_config, false);

    // --- Compute actual Hyrax proof ---
    let mut hyrax_provable_circuit = circuit_desc.gen_hyrax_provable_circuit().unwrap();
    let hyrax_verifiable_circuit = hyrax_provable_circuit._gen_hyrax_verifiable_circuit(); // Testing only
    let mut prover_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    let (proof, proof_config) = perform_function_under_prover_config!(
        |w, x, y, z| hyrax_provable_circuit.prove(w, x, y, z), // Thanks Chat!
        &gkr_circuit_prover_config,
        &committer,
        &mut blinding_rng,
        converter,
        &mut prover_transcript
    );

    // Create new transcript for holistic verifier to follow along with
    let mut verifier_transcript: ECTranscript<Bn256Point, PoseidonSponge<Base>> =
        ECTranscript::new("modulus modulus modulus modulus modulus");

    perform_function_under_verifier_config!(
        verify_hyrax_proof,
        &gkr_circuit_verifier_config,
        &proof,
        &hyrax_verifiable_circuit,
        &committer,
        &mut verifier_transcript,
        &proof_config
    );
}

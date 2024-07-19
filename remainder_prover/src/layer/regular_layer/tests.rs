use ark_std::test_rng;
use remainder_shared_types::{
    transcript::{
        poseidon_transcript::PoseidonSponge, test_transcript::TestSponge, TranscriptReader,
        TranscriptWriter,
    },
    Fr,
};

use crate::{
    claims::Claim,
    expression::{circuit_expr::CircuitExpr, generic_expr::Expression, prover_expr::ProverExpr},
    layer::{CircuitLayer, Layer, LayerId},
    mle::dense::DenseMle,
};

use super::{CircuitRegularLayer, RegularLayer};

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_product() {
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_ref_1 = mle_new;
    let mle_ref_2 = mle_2;

    let mut expression = Expression::<Fr, ProverExpr>::products(vec![mle_ref_1, mle_ref_2]);
    let claim = crate::sumcheck::tests::get_dummy_expression_eval(&expression, &mut rng);
    dbg!(&claim);

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove_rounds(claim.clone(), &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    dbg!(&transcript_raw);
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);
    let circuit_expression = expression.transform_to_circuit_expression().unwrap();
    dbg!(&circuit_expression);
    let mut verifier_layer = CircuitRegularLayer::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(claim, &mut transcript)
        .unwrap();
}

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_sum() {
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_ref_1 = mle_new;
    let mle_ref_2 = mle_2;

    let lhs = Expression::<Fr, ProverExpr>::mle(mle_ref_1);
    let rhs = Expression::<Fr, ProverExpr>::mle(mle_ref_2);
    let mut expression = Expression::<Fr, ProverExpr>::sum(lhs, rhs);
    // let claim = crate::sumcheck::tests::get_dummy_expression_eval(&expression, &mut rng);
    let claim = Claim::<Fr>::new(vec![Fr::from(2), Fr::from(3)], Fr::from(10));
    dbg!(&claim);

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove_rounds(claim.clone(), &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    dbg!(&transcript_raw);
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);
    let circuit_expression = expression.transform_to_circuit_expression().unwrap();
    dbg!(&circuit_expression);
    let mut verifier_layer = CircuitRegularLayer::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(claim, &mut transcript)
        .unwrap();
}

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_selector() {
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_ref_1 = mle_new;
    let mle_ref_2 = mle_2;

    let lhs = Expression::<Fr, ProverExpr>::mle(mle_ref_1);
    let rhs = Expression::<Fr, ProverExpr>::mle(mle_ref_2);
    let mut expression = rhs.concat_expr(lhs);
    // let claim = crate::sumcheck::tests::get_dummy_expression_eval(&expression, &mut rng);
    dbg!(&expression);
    let claim = Claim::<Fr>::new(vec![Fr::from(4), Fr::from(2), Fr::from(3)], Fr::from(33));

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove_rounds(claim.clone(), &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    dbg!(&transcript_raw);
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);
    let circuit_expression = expression.transform_to_circuit_expression().unwrap();
    dbg!(&circuit_expression);
    let mut verifier_layer = CircuitRegularLayer::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(claim, &mut transcript)
        .unwrap();
}

#[test]
fn regular_layer_test_prove_verify_complex() {
    let mut rng = test_rng();

    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)],
        LayerId::Input(0),
    );
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)],
        LayerId::Input(0),
    );

    let leaf_mle_1 = Expression::<Fr, ProverExpr>::mle(mle_1.clone());
    let leaf_mle_2 = Expression::<Fr, ProverExpr>::mle(mle_2.clone());
    let sum = Expression::<Fr, ProverExpr>::sum(leaf_mle_2, leaf_mle_1);

    let prod = Expression::<Fr, ProverExpr>::products(vec![mle_1.clone(), mle_2.clone()]);

    let mut root = sum.concat_expr(prod);
    dbg!(&root);

    let claim = Claim::<Fr>::new(vec![Fr::from(4), Fr::from(2), Fr::from(3)], Fr::from(37));

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), root.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove_rounds(claim.clone(), &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    dbg!(&transcript_raw);
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    root.index_mle_indices(0);
    let circuit_expression = root.transform_to_circuit_expression().unwrap();
    dbg!(&circuit_expression);
    let mut verifier_layer = CircuitRegularLayer::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(claim, &mut transcript)
        .unwrap();
}

// #[test]
// /// Testing of the ability of `RegularLayer` to yield Claims after proving
// fn regular_layer_test_yield_claims() {
//     todo!()
// }

// #[test]
// /// Testing of `RegularLayer`'s YieldWlxEvals implementation
// fn regular_layer_test_get_wlx_evals() {
//     todo!()
// }

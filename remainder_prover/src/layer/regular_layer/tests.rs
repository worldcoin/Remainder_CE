use ark_std::test_rng;
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonSponge, TranscriptReader, TranscriptWriter},
    Fr,
};

use crate::{
    claims::RawClaim,
    expression::{
        circuit_expr::ExprDescription, generic_expr::Expression, prover_expr::ProverExpr,
    },
    layer::{Layer, LayerDescription, LayerId},
    mle::{dense::DenseMle, mle_description::MleDescription, Mle},
};

use super::{RegularLayer, RegularLayerDescription};

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_product() {
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_1 = mle_new;
    let mle_2 = mle_2;

    let circuit_mle_1 = MleDescription::new(LayerId::Input(0), mle_1.mle_indices());
    let circuit_mle_2 = MleDescription::new(LayerId::Input(0), mle_2.mle_indices());
    let mut circuit_expression =
        Expression::<Fr, ExprDescription>::products(vec![circuit_mle_1, circuit_mle_2]);
    circuit_expression.index_mle_vars(0);

    let mut expression = Expression::<Fr, ProverExpr>::products(vec![mle_1, mle_2]);
    let claim = crate::sumcheck::tests::get_dummy_expression_eval(&expression, &mut rng);

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove(&[&claim], &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);

    let verifier_layer = RegularLayerDescription::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(&[&claim], &mut transcript)
        .unwrap();
}

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_sum() {
    let mle_vec = vec![Fr::from(2), Fr::from(1), Fr::from(3), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(1), Fr::from(5), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_1 = mle_new;
    let mle_2 = mle_2;

    let circuit_mle_1 = MleDescription::new(LayerId::Input(0), mle_1.mle_indices());
    let circuit_mle_2 = MleDescription::new(LayerId::Input(0), mle_2.mle_indices());
    let mut circuit_expression = Expression::<Fr, ExprDescription>::sum(
        Expression::from_mle_desc(circuit_mle_1),
        Expression::from_mle_desc(circuit_mle_2),
    );
    circuit_expression.index_mle_vars(0);

    let lhs = Expression::<Fr, ProverExpr>::mle(mle_1);
    let rhs = Expression::<Fr, ProverExpr>::mle(mle_2);
    let mut expression = Expression::<Fr, ProverExpr>::sum(lhs, rhs);
    let claim = RawClaim::<Fr>::new(vec![Fr::from(2), Fr::from(3)], Fr::from(10));

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove(&[&claim], &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);
    let verifier_layer = RegularLayerDescription::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(&[&claim], &mut transcript)
        .unwrap();
}

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify_selector() {
    let mle_vec = vec![Fr::from(2), Fr::from(1), Fr::from(3), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mle_v2 = vec![Fr::from(1), Fr::from(1), Fr::from(5), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0));

    let mle_1 = mle_new;
    let mle_2 = mle_2;

    let circuit_mle_1 = MleDescription::new(LayerId::Input(0), mle_1.mle_indices());
    let circuit_mle_2 = MleDescription::new(LayerId::Input(0), mle_2.mle_indices());
    let mut circuit_expression = Expression::<Fr, ExprDescription>::selectors(vec![
        Expression::from_mle_desc(circuit_mle_1),
        Expression::from_mle_desc(circuit_mle_2),
    ]);
    circuit_expression.index_mle_vars(0);

    let lhs = Expression::<Fr, ProverExpr>::mle(mle_1);
    let rhs = Expression::<Fr, ProverExpr>::mle(mle_2);
    let mut expression = lhs.select(rhs);
    let claim = RawClaim::<Fr>::new(vec![Fr::from(4), Fr::from(2), Fr::from(3)], Fr::from(33));

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove(&[&claim], &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    expression.index_mle_indices(0);

    let verifier_layer = RegularLayerDescription::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(&[&claim], &mut transcript)
        .unwrap();
}

#[test]
fn regular_layer_test_prove_verify_complex() {
    let mle_1: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(1), Fr::from(3), Fr::from(2)],
        LayerId::Input(0),
    );
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(1), Fr::from(5), Fr::from(5)],
        LayerId::Input(0),
    );

    let leaf_mle_1 = Expression::<Fr, ProverExpr>::mle(mle_1.clone());
    let leaf_mle_2 = Expression::<Fr, ProverExpr>::mle(mle_2.clone());

    let circuit_mle_1 = MleDescription::new(LayerId::Input(0), mle_1.mle_indices());
    let circuit_mle_2 = MleDescription::new(LayerId::Input(0), mle_2.mle_indices());
    let mut circuit_expression = Expression::<Fr, ExprDescription>::selectors(vec![
        Expression::<Fr, ExprDescription>::products(vec![
            circuit_mle_1.clone(),
            circuit_mle_2.clone(),
        ]),
        Expression::from_mle_desc(circuit_mle_2) + Expression::from_mle_desc(circuit_mle_1),
    ]);
    circuit_expression.index_mle_vars(0);
    let sum = Expression::<Fr, ProverExpr>::sum(leaf_mle_2, leaf_mle_1);

    let prod = Expression::<Fr, ProverExpr>::products(vec![mle_1.clone(), mle_2.clone()]);

    let mut root = prod.select(sum);

    let claim = RawClaim::<Fr>::new(vec![Fr::from(4), Fr::from(2), Fr::from(3)], Fr::from(37));

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), root.clone());

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    layer.prove(&[&claim], &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    root.index_mle_indices(0);
    let verifier_layer = RegularLayerDescription::new_raw(LayerId::Layer(0), circuit_expression);

    verifier_layer
        .verify_rounds(&[&claim], &mut transcript)
        .unwrap();
}

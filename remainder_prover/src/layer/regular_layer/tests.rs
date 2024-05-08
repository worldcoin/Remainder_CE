use ark_std::test_rng;
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonSponge, TranscriptReader, TranscriptWriter},
    Fr,
};

use crate::{
    expression::generic_expr::Expression,
    layer::{Layer, LayerId},
    mle::dense::DenseMle,
};

use super::RegularLayer;

#[test]
/// E2E test of Proving/Verifying a `RegularLayer`
fn regular_layer_test_prove_verify() {
    let mut rng = test_rng();
    let mle_vec = vec![Fr::from(2), Fr::from(3), Fr::from(1), Fr::from(2)];

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mle_v2 = vec![Fr::from(1), Fr::from(5), Fr::from(1), Fr::from(5)];
    let mle_2: DenseMle<Fr> = DenseMle::new_from_raw(mle_v2, LayerId::Input(0), None);

    let mle_ref_1 = mle_new.mle_ref();
    let mle_ref_2 = mle_2.mle_ref();

    let expression = Expression::products(vec![mle_ref_1, mle_ref_2]);
    let claim = crate::sumcheck::tests::get_dummy_expression_eval(&expression, &mut rng);

    let mut layer = RegularLayer::new_raw(crate::layer::LayerId::Layer(0), expression);

    let mut transcript = TranscriptWriter::<_, PoseidonSponge<_>>::new("Regular Layer Test");

    let proof = layer.prove_rounds(claim.clone(), &mut transcript).unwrap();

    let transcript_raw = transcript.get_transcript();
    let mut transcript = TranscriptReader::<_, PoseidonSponge<_>>::new(transcript_raw);

    layer.verify_rounds(claim, proof, &mut transcript).unwrap();
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

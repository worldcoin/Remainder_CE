//! Trait for dealing with InputLayer

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::*;
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use thiserror::Error;
use tracing::instrument;
pub mod combine_input_layers;
pub mod enum_input_layer;
pub mod ligero_input_layer;
pub mod public_input_layer;
pub mod random_input_layer;
mod tests;

use crate::{
    layer::{
        claims::{get_num_wlx_evaluations, Claim, ClaimError, ClaimGroup, ENABLE_PRE_FIX},
        combine_mle_refs::pre_fix_mle_refs,
        LayerId,
    },
    mle::{dense::DenseMle, mle_enum::MleEnum, MleIndex, MleRef},
    prover::ENABLE_OPTIMIZATION,
    sumcheck::evaluate_at_a_point,
};

use self::enum_input_layer::InputLayerEnum;

use ark_std::{end_timer, start_timer};

#[derive(Error, Clone, Debug)]
pub enum InputLayerError {
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
    #[error("failed to verify random input layer")]
    RandomInputVerificationFailed,
    #[error("Error during interaction with the transcript.")]
    TranscriptError(TranscriptReaderError),
    #[error("Challenge or consumed element did not match the expected value.")]
    TranscriptMatchError,
}

use log::{debug, info, trace, warn};
///Trait for dealing with the InputLayer
pub trait InputLayer<F: FieldExt> {
    type Sponge: TranscriptSponge<F>;
    type Commitment;
    type OpeningProof;

    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    );

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, Self::Sponge>,
    ) -> Result<(), InputLayerError>;

    fn open(
        &self,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, InputLayerError>;

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript_reader: &mut TranscriptReader<F, Self::Sponge>,
    ) -> Result<(), InputLayerError>;

    fn layer_id(&self) -> &LayerId;

    fn get_padded_mle(&self) -> DenseMle<F, F>;

    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    #[instrument(skip_all, err, level = "debug")]
    fn compute_claim_wlx(&self, claims: &ClaimGroup<F>) -> Result<Vec<F>, ClaimError> {
        let prep_timer = start_timer!(|| "Claim wlx prep");
        let mut mle_ref = self.get_padded_mle().clone().mle_ref();
        end_timer!(prep_timer);
        info!(
            "Wlx MLE len: {}",
            mle_ref.current_mle.get_evals_vector().len()
        );
        let num_claims = claims.get_num_claims();
        let claim_vecs = claims.get_claim_points_matrix();
        let claimed_vals = claims.get_results();
        let num_idx = claims.get_num_vars();

        //fix variable hella times
        //evaluate expr on the mutated expr

        // get the number of evaluations
        mle_ref.index_mle_indices(0);
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);
        let chal_point = &claim_vecs[0];

        if ENABLE_PRE_FIX {
            if common_idx.is_some() {
                let common_idx = common_idx.unwrap();
                common_idx.iter().for_each(|chal_idx| {
                    if let MleIndex::IndexedBit(idx_bit_num) = mle_ref.mle_indices()[*chal_idx] {
                        mle_ref.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
                    }
                });
            }
        }

        debug!("Evaluating {num_evals} times.");

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            // let next_evals: Vec<F> = (num_claims..num_evals).into_iter()
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    // let new_chal: Vec<F> = (0..num_idx).into_iter()
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            // let evals: Vec<F> = (&claim_vecs).into_iter()
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                let mut fix_mle = mle_ref.clone();
                {
                    new_chal.into_iter().enumerate().for_each(|(idx, chal)| {
                        if let MleIndex::IndexedBit(idx_num) = fix_mle.mle_indices()[idx] {
                            fix_mle.fix_variable(idx_num, chal);
                        }
                    });
                    fix_mle.current_mle[0]
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        debug!("Returning evals:\n{:#?} ", wlx_evals);
        Ok(wlx_evals)
    }

    /// Computes `l(r^{\star})`
    #[instrument(skip_all, err, level = "debug")]
    fn compute_aggregated_challenges(
        &self,
        claims: &ClaimGroup<F>,
        rstar: F,
    ) -> Result<Vec<F>, ClaimError> {
        let claim_vecs = claims.get_claim_points_matrix();

        if claims.is_empty() {
            return Err(ClaimError::ClaimAggroError);
        }
        let num_idx = claim_vecs[0].len();

        // get the claim (r = l(r*))
        let r: Vec<F> = cfg_into_iter!(0..num_idx)
            .map(|idx| {
                let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                    .map(|claim| claim[idx])
                    .collect();
                evaluate_at_a_point(&evals, rstar).unwrap()
            })
            .collect();

        Ok(r)
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Sponge>;
}

///Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    ///Creates a new InputLayer from a flat mle
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self;
}

//! Trait for dealing with InputLayer

use ark_std::cfg_into_iter;

use rayon::prelude::*;
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An enum which represents which type of input layer we are working with.
pub mod enum_input_layer;
/// An input layer in which the input data is committed to using the Ligero PCS.
pub mod ligero_input_layer;
/// An input layer which requires no commitment and is openly evaluated at the random point.
pub mod public_input_layer;
/// An input layer in order to generate random challenges for Fiat-Shamir.
pub mod random_input_layer;

#[cfg(test)]
mod tests;

use crate::{
    claims::{wlx_eval::get_num_wlx_evaluations, Claim},
    layer::LayerId,
    mle::{dense::DenseMle, mle_enum::MleEnum, MleIndex, MleRef},
    sumcheck::evaluate_at_a_point,
};

use ark_std::{end_timer, start_timer};

#[derive(Error, Clone, Debug)]
/// The errors which can be encountered when constructing an input layer.
pub enum InputLayerError {
    /// Commitments can only be opened if they exist. If the commitment is not generated
    /// but is attempted to be opened, this error will be thrown.
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    /// This is when the public input layer polynomial evaluated at a random point
    /// does not equal the claimed value.
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
    /// This is when the random input layer evaluated at a random point does not
    /// equal the claimed value.
    #[error("failed to verify random input layer")]
    RandomInputVerificationFailed,
    /// This is when there is an error when trying to squeeze or add elements to the transcript.
    #[error("Error during interaction with the transcript.")]
    TranscriptError(TranscriptReaderError),
    /// This is thrown when the transcript squeezed value for the verifier does not
    /// match what the prover squeezed for the same point.
    #[error("Challenge or consumed element did not match the expected value.")]
    TranscriptMatchError,
}

use log::{debug, info};
/// The InputLayer trait in which the evaluation proof, commitment, and proof/verification
/// process takes place for input layers.
pub trait InputLayer<F: FieldExt> {
    /// The struct that contains the commitment to the contents of the input_layer.
    type Commitment: Serialize + for<'a> Deserialize<'a>;

    /// The struct that contains the opening proof.
    type OpeningProof: Serialize + for<'a> Deserialize<'a>;

    /// Generates a commitment
    ///
    /// Can mutate self to cache useful information.
    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    /// Appends the commitment to the F-S Transcript.
    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    );

    /// Appends the commitment to the F-S Transcript.
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError>;

    /// Generates a proof of polynomial evaluation at the point
    /// in the `Claim`.
    ///
    /// Appends any communication to the transcript.
    fn open(
        &self,
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, InputLayerError>;

    /// Verifies the evaluation at the point in the `Claim` relative to the
    /// polynomial commitment using the opening proof.
    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError>;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> &LayerId;

    /// Returns the contents of this `InputLayer` as an
    /// owned `DenseMle`.
    fn get_padded_mle(&self) -> DenseMle<F, F>;
}

/// Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs.
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    /// Creates a new InputLayer from a flat mle.
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self;
}

/// Computes the V_d(l(x)) evaluations for the input layer V_d.
fn get_wlx_evaluations_helper<F: FieldExt>(
    layer: &impl InputLayer<F>,
    claim_vecs: &[Vec<F>],
    claimed_vals: &[F],
    _claimed_mles: Vec<MleEnum<F>>,
    num_claims: usize,
    num_idx: usize,
) -> Result<Vec<F>, crate::claims::ClaimError> {
    let prep_timer = start_timer!(|| "Claim wlx prep");
    let mut mle_ref = layer.get_padded_mle().mle_ref();
    end_timer!(prep_timer);
    info!(
        "Wlx MLE len: {}",
        mle_ref.current_mle.get_evals_vector().len()
    );

    mle_ref.index_mle_indices(0);
    // Get the number of evaluations needed depending on the claim vectors.
    let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);
    let chal_point = &claim_vecs[0];

    if let Some(common_idx) = common_idx {
        common_idx.iter().for_each(|chal_idx| {
            if let MleIndex::IndexedBit(idx_bit_num) = mle_ref.mle_indices()[*chal_idx] {
                mle_ref.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
            }
        });
    }

    debug!("Evaluating {num_evals} times.");

    // We already have the first #claims evaluations, get the next num_evals - #claims evaluations.
    let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
        .map(|idx| {
            // Get the challenge l(idx).
            let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                .map(|claim_idx| {
                    let evals: Vec<F> = cfg_into_iter!(claim_vecs)
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

    // Concat this with the first k evaluations from the claims to get num_evals evaluations.
    let mut wlx_evals = claimed_vals.to_vec();
    wlx_evals.extend(&next_evals);
    debug!("Returning evals:\n{:#?} ", wlx_evals);
    Ok(wlx_evals)
}

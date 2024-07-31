//! Trait for dealing with InputLayer

use ark_std::cfg_into_iter;
use remainder_shared_types::transcript::{ProverTranscript, VerifierTranscript};

use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use rayon::prelude::*;
use remainder_shared_types::{transcript::TranscriptReaderError, FieldExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{claims::Claim, layer::LayerId};

/// An enum which represents which type of input layer we are working with.
pub mod enum_input_layer;
/// An input layer in which the input data is committed to using the Ligero PCS.
pub mod ligero_input_layer;
/// An input layer which requires no commitment and is openly evaluated at the random point.
pub mod public_input_layer;
/// An input layer in order to generate random challenges for Fiat-Shamir.
pub mod random_input_layer;

/// An input layer in order to distinguish Hyrax input layers from others.
/// NOTE: this input layer is just a placeholder to convert from impl [GKRCircuit]s to [HyraxCircuit]s, but
/// the functionality should NOT be used in just a regular [GKRCircuit].
pub mod hyrax_placeholder_input_layer;

/// An input layer in order to distinguish Hyrax input layers with precommits from others.
/// NOTE: this input layer is just a placeholder to convert from impl [GKRCircuit]s to [HyraxCircuit]s, but
/// the functionality should NOT be used in just a regular [GKRCircuit].
pub mod hyrax_precommit_placeholder_input_layer;

#[cfg(test)]
mod tests;

use crate::{
    claims::wlx_eval::get_num_wlx_evaluations, mle::mle_enum::MleEnum,
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
    TranscriptError(#[from] TranscriptReaderError),
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
    type Commitment: Serialize + for<'a> Deserialize<'a> + core::fmt::Debug;

    /// The Verifier Key representation for this input layer.
    type VerifierInputLayer: VerifierInputLayer<F, Commitment = Self::Commitment>
        + Serialize
        + for<'a> Deserialize<'a>
        + core::fmt::Debug;

    /// Returns the circuit description of this layer for the verifier.
    fn into_verifier_input_layer(&self) -> Self::VerifierInputLayer;

    /// Generates and returns a commitment.
    /// May also store it internally.
    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    /// Appends the commitment to the F-S Transcript.
    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut impl ProverTranscript<F>,
    );

    /// Generates a proof of polynomial evaluation at the point
    /// in the `Claim`.
    ///
    /// Appends any communication to the transcript.
    fn open(
        &self,
        transcript_writer: &mut impl ProverTranscript<F>,
        claim: Claim<F>,
    ) -> Result<(), InputLayerError>;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> LayerId;

    /// Returns the contents of this `InputLayer` as an
    /// owned `DenseMle`.
    fn get_padded_mle(&self) -> DenseMle<F>;
}

pub trait VerifierInputLayer<F: FieldExt> {
    /// The struct that contains the commitment to the contents of the input_layer.
    type Commitment: Serialize + for<'a> Deserialize<'a> + core::fmt::Debug;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> LayerId;

    /// Read the commitment off of the transcript.
    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError>;

    /// Verifies the evaluation at the point in the `Claim` relative to the
    /// polynomial commitment using the opening proof in the transcript.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError>;
}

/// Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs.
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    /// Creates a new InputLayer from a flat mle.
    fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self;
}

/// Computes the V_d(l(x)) evaluations for the input layer V_d.
fn get_wlx_evaluations_helper<F: FieldExt>(
    mut mle_ref: MultilinearExtension<F>,
    claim_vecs: &[Vec<F>],
    claimed_vals: &[F],
    _claimed_mles: Vec<MleEnum<F>>,
    num_claims: usize,
    num_idx: usize,
) -> Result<Vec<F>, crate::claims::ClaimError> {
    let prep_timer = start_timer!(|| "Claim wlx prep");
    end_timer!(prep_timer);
    info!("Wlx MLE len: {}", mle_ref.get_evals_vector().len());
    // Get the number of evaluations needed depending on the claim vectors.
    let (num_evals, common_idx, non_common_idx) = get_num_wlx_evaluations(claim_vecs);
    let chal_point = &claim_vecs[0];

    dbg!(&mle_ref);
    dbg!(&claim_vecs);
    dbg!(&common_idx);
    if let Some(common_idx) = common_idx {
        common_idx.iter().for_each(|chal_idx| {
            mle_ref.fix_variable_at_index(*chal_idx, chal_point[*chal_idx]);
        });
    }
    dbg!(&mle_ref);
    debug!("Evaluating {num_evals} times.");

    // We already have the first #claims evaluations, get the next num_evals - #claims evaluations.
    let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
        .map(|idx| {
            // Get the challenge l(idx).
            let new_chal: Vec<F> = cfg_into_iter!(non_common_idx.clone())
                .map(|claim_idx| {
                    let evals: Vec<F> = cfg_into_iter!(claim_vecs)
                        .map(|claim| claim[claim_idx])
                        .collect();
                    evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                })
                .collect();

            let mut fix_mle = mle_ref.clone();
            dbg!(&fix_mle);
            dbg!(&new_chal);
            {
                new_chal.into_iter().for_each(|chal| {
                    fix_mle.fix_variable(chal);
                });
                fix_mle.get_evals()[0]
            }
        })
        .collect();

    // Concat this with the first k evaluations from the claims to get num_evals evaluations.
    let mut wlx_evals = claimed_vals.to_vec();
    wlx_evals.extend(&next_evals);
    debug!("Returning evals:\n{:#?} ", wlx_evals);
    Ok(wlx_evals)
}

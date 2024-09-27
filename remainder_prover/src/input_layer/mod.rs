//! Trait for dealing with InputLayer

use ark_std::cfg_into_iter;
use enum_input_layer::InputLayerEnum;
use itertools::Itertools;
use remainder_ligero::ligero_structs::LigeroCommit;
use remainder_ligero::poseidon_ligero::PoseidonSpongeHasher;
use remainder_shared_types::transcript::{ProverTranscript, VerifierTranscript};

use crate::layer::regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION;
use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use remainder_shared_types::{transcript::TranscriptReaderError, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{claims::Claim, layer::LayerId};

/// An enum which represents which type of input layer we are working with.
pub mod enum_input_layer;
/// An input layer in order to generate random challenges for Fiat-Shamir.
pub mod fiat_shamir_challenge;
/// The circuit description struct for the input layer where the data is committed to using the Hyrax PCS.
pub mod hyrax_input_layer;
/// An input layer in which the input data is committed to using the Ligero PCS.
pub mod ligero_input_layer;
/// An input layer which requires no commitment and is openly evaluated at the random point.
pub mod public_input_layer;

use crate::{
    claims::wlx_eval::get_num_wlx_evaluations, mle::mle_enum::MleEnum,
    sumcheck::evaluate_at_a_point,
};

use ark_std::{end_timer, start_timer};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(bound = "F: Field")]
/// An enum for representing the different types of commitments for each type
/// of input layer.
pub enum CommitmentEnum<F: Field> {
    /// The commitment for a [LigeroInputLayer].
    LigeroCommitment(LigeroCommit<PoseidonSpongeHasher<F>, F>),
    /// The commitment for a [PublicInputLayer]
    PublicCommitment(Vec<F>),
}

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
pub trait InputLayer<F: Field> {
    /// The struct that contains the commitment to the contents of the input_layer in the prover's view.
    type ProverCommitment: Serialize + for<'a> Deserialize<'a> + core::fmt::Debug;

    /// The struct that contains the commitment to the contents of the input_layer in the verifier's view.
    /// This is what should be added to transcript, because
    type VerifierCommitment: Serialize + for<'a> Deserialize<'a> + core::fmt::Debug;

    /// Generates and returns a commitment.
    /// May also store it internally.
    fn commit(&mut self) -> Result<Self::VerifierCommitment, InputLayerError>;

    /// Appends the commitment to the F-S Transcript.
    fn append_commitment_to_transcript(
        commitment: &Self::VerifierCommitment,
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

/// The trait representing methods necessary for the circuit description of an input layer.
pub trait CircuitInputLayer<F: Field> {
    /// The struct that contains the commitment to the contents of the input_layer.
    type Commitment: Serialize + for<'a> Deserialize<'a> + core::fmt::Debug;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> LayerId;

    /// Read the commitment off of the transcript.
    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError>;

    /// Convert a circuit input layer into a prover input layer.
    fn convert_into_prover_input_layer(
        &self,
        mle: MultilinearExtension<F>,
        precommit: &Option<CommitmentEnum<F>>,
    ) -> InputLayerEnum<F>;

    /// Verifies the evaluation at the point in the `Claim` relative to the
    /// polynomial commitment using the opening proof in the transcript.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError>;
}

/// Computes the V_d(l(x)) evaluations for the input layer V_d.
fn get_wlx_evaluations_helper<F: Field>(
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
    let (num_evals, common_idx, non_common_idx) = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION
    {
        let (num_evals, common_idx, non_common_idx) = get_num_wlx_evaluations(claim_vecs);
        (num_evals, common_idx, non_common_idx)
    } else {
        let num_evals = ((num_claims - 1) * num_idx) + 1;
        let common_idx = None;
        let non_common_idx = (0..num_idx).collect_vec();
        (num_evals, common_idx, non_common_idx)
    };

    let chal_point = &claim_vecs[0];

    if let Some(common_idx) = common_idx {
        let mut common_idx_sorted = common_idx.clone();
        common_idx_sorted.sort();
        common_idx_sorted
            .iter()
            .enumerate()
            .for_each(|(offset_idx, chal_idx)| {
                mle_ref.fix_variable_at_index(*chal_idx - offset_idx, chal_point[*chal_idx]);
            });
    }
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

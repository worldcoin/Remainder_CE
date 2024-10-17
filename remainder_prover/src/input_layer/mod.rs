//! Trait for dealing with InputLayer

use enum_input_layer::InputLayerEnum;
use remainder_ligero::ligero_structs::LigeroCommit;
use remainder_ligero::poseidon_ligero::PoseidonSpongeHasher;
use remainder_shared_types::transcript::{ProverTranscript, VerifierTranscript};

use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use remainder_shared_types::{transcript::TranscriptReaderError, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{claims::RawClaim, layer::LayerId};

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
        claim: RawClaim<F>,
    ) -> Result<(), InputLayerError>;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> LayerId;

    /// Returns the contents of this `InputLayer` as an
    /// owned `DenseMle`.
    fn get_padded_mle(&self) -> DenseMle<F>;
}

/// The trait representing methods necessary for the circuit description of an input layer.
pub trait InputLayerDescription<F: Field> {
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
        claim: RawClaim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError>;
}

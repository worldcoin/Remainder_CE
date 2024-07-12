use serde::{Deserialize, Serialize};

use crate::{
    claims::Claim,
    layer::LayerId,
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};

/// The InputLayer trait in which the evaluation proof, commitment, and proof/verification
/// process takes place for input layers.
pub trait InputLayer<F: FieldExt> {
    /// The struct that contains the commitment to the contents of the input_layer.
    type Commitment: Serialize + for<'a> Deserialize<'a>;

    /// The struct that contains the opening proof.
    type OpeningProof: Serialize + for<'a> Deserialize<'a>;

    type Error: std::error::Error;

    /// Generates a commitment
    ///
    /// Can mutate self to cache useful information.
    fn commit(&mut self) -> Result<Self::Commitment, Self::Error>;

    /// Appends the commitment to the F-S Transcript.
    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut impl ProverTranscript<F>,
    );

    /// Appends the commitment to the F-S Transcript.
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<(), Self::Error>;

    /// Generates a proof of polynomial evaluation at the point
    /// in the `Claim`.
    ///
    /// Appends any communication to the transcript.
    fn open(
        &self,
        transcript: &mut impl ProverTranscript<F>,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, Self::Error>;

    /// Verifies the evaluation at the point in the `Claim` relative to the
    /// polynomial commitment using the opening proof.
    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<(), Self::Error>;

    /// Returns the `LayerId` of this layer.
    fn layer_id(&self) -> &LayerId;
}

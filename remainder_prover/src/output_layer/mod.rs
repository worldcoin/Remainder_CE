//! Traits and implementations for GKR Output Layers.

use remainder_shared_types::{
    transcript::{
        ProverTranscript, TranscriptReader, TranscriptReaderError, TranscriptSponge,
        TranscriptWriter, VerifierTranscript,
    },
    FieldExt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    expression::{generic_expr::Expression, verifier_expr::VerifierExpr},
    layer::{LayerError, LayerId},
};

// The default implementation of an Output layer which is an MLE.
pub mod mle_output_layer;

// Unit tests for Output Layers.
#[cfg(test)]
pub mod tests;

/// Errors to do with working with a type implementing [OutputLayer].
#[derive(Error, Debug, Clone)]
pub enum OutputLayerError {
    /// Expected fully-bound MLE.
    #[error("Expected fully-bound MLE")]
    MleNotFullyBound,
}

/// Errors to do with working with a type implementing [VerifierOutputLayer].
#[derive(Error, Debug, Clone)]
pub enum VerifierOutputLayerError {
    /// Prover sent a non-zero value for a ZeroMle.
    #[error("Prover sent a non-zero value for a ZeroMle")]
    NonZeroEvalForZeroMle,

    /// Transcript Reader Error during verification.
    #[error("Transcript Reader Error: {:0}", _0)]
    TranscripError(#[from] TranscriptReaderError),
}

/// The interface of a Prover's Output Layer.
/// Output layers are "virtual layers" in the sense that they are not assigned a
/// separate [LayerId]. Instead they are associated with the ID of an existing
/// intermediate/input layer on which they generate claims for.
pub trait OutputLayer<F: FieldExt> {
    /// The associated type for the circuit-description analogue of this Ouput
    /// Layer.
    type CircuitOutputLayer: CircuitOutputLayer<F>
        + Serialize
        + for<'a> Deserialize<'a>
        + core::fmt::Debug;

    /// Returns the [LayerId] of the intermediate/input layer that this output
    /// layer is associated with.
    fn layer_id(&self) -> LayerId;

    /// Append the original MLE representation to the transcript.
    fn append_mle_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<F>);

    /// Fix the variables of this output layer to random challenges sampled
    /// from the transcript.
    fn fix_layer(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError>;

    /// Return the Circuit description for this Output Layer to be used
    /// by the verifier.
    /// Should be called before any other method that mutates `self`!
    fn into_circuit_output_layer(&self) -> Self::CircuitOutputLayer;
}

/// The interface for the circuit description counterpart of an Output Layer.
pub trait CircuitOutputLayer<F: FieldExt> {
    /// The associated type used by the verifier for manipulating an Ouput
    /// Layer.
    type VerifierOutputLayer: VerifierOutputLayer<F> + Serialize + for<'a> Deserialize<'a>;

    /// Returns the [LayerId] of the intermediate/input layer that his output
    /// layer is associated with.
    fn layer_id(&self) -> LayerId;

    /*
    /// Retrieve the original MLE representation from the transcript.
    fn retrieve_mle_from_transcript(
        &mut self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), VerifierOutputLayerError>;
    */

    /// Retrieve the MLE evaluations from the transcript and fix the variables
    /// of this output layer to random challenges sampled from the transcript.
    /// Returns a description of the layer ready to be used by the verifier.
    fn retrieve_mle_from_transcript_and_fix_layer(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierOutputLayer, VerifierOutputLayerError>;
}

/// The interface for the verifier's counterpart of an Output Layer.
/// This trait should be able to yield claims!
pub trait VerifierOutputLayer<F: FieldExt> {
    /// Returns the [LayerId] of the intermediate/input layer that his output
    /// layer is associated with.
    fn layer_id(&self) -> LayerId;
}

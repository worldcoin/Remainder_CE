//! Traits and implementations for GKR Output Layers.

use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::layer::{LayerError, LayerId};

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
    TranscriptError(#[from] TranscriptReaderError),
}
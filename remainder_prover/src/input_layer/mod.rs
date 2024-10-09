//! Trait for dealing with InputLayer

use crate::mle::evals::MultilinearExtension;
use remainder_shared_types::{transcript::TranscriptReaderError, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::layer::LayerId;

/// An input layer in order to generate random challenges for Fiat-Shamir.
pub mod fiat_shamir_challenge;
/// An input layer in which the input data is committed to using the Ligero PCS.
pub mod ligero_input_layer;
/// An input layer which requires no commitment and is openly evaluated at the random point.
pub mod public_input_layer;

/// The prover's view of an input layer during circuit proving (undifferentiated as to type).
/// Note that, being undifferentiated, no functions for adding values or commitments to the transcript are provided.
#[derive(Debug, Clone)]
pub struct InputLayer<F: Field> {
    /// The MLE for the input layer.
    pub mle: MultilinearExtension<F>,
    /// The layer ID of the input layer.
    pub layer_id: LayerId,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
/// The verifier's view of an input layer during circuit proving, containing
/// the shape information of this input layer.
pub struct InputLayerDescription {
    /// The layer ID of the input layer.
    pub layer_id: LayerId,
    /// The number of variables in the input layer.
    pub num_vars: usize,
}
//! Trait for dealing with InputLayer

use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::transcript::ProverTranscript;
use tracing::{debug, info};

use crate::claims::wlx_eval::YieldWLXEvals;
use crate::layer::regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION;
use crate::mle::dense::DenseMle;
use crate::mle::evals::MultilinearExtension;
use remainder_shared_types::{transcript::TranscriptReaderError, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{claims::Claim, layer::LayerId};

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

use crate::{claims::wlx_eval::get_num_wlx_evaluations, sumcheck::evaluate_at_a_point};

use ark_std::{end_timer, start_timer};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Error, Clone, Debug)]
/// The errors which can be encountered when constructing an input layer.
pub enum InputLayerError {
    /// Commitments can only be opened if they exist. If the commitment is not generated
    /// but is attempted to be opened, this error will be thrown.
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    /// This is when there is an error when trying to squeeze or add elements to the transcript.
    #[error("Error during interaction with the transcript.")]
    TranscriptError(#[from] TranscriptReaderError),
    /// This is thrown when the transcript squeezed value for the verifier does not
    /// match what the prover squeezed for the same point.
    #[error("Challenge or consumed element did not match the expected value.")]
    TranscriptMatchError,
}
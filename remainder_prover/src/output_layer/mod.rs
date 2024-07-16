//! Traits and implementations for GKR Output Layers.

use remainder_shared_types::{
    transcript::{TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::layer::{LayerError, LayerId};

/// The default implementation of an Output layer which is an MLE.
pub mod mle_output_layer;

/// The interface of a Prover's Output Layer.
/// Output layers are "virtual layers" in the sense that they are not assigned a
/// separate [LayerId]. Instead they are associated with the ID of an existing
/// intermediate/input layer on which they generate claims for.
pub trait OutputLayer<F: FieldExt> {
    /// The associated type for the verifier analogue of this Ouput Layer.
    type VerifierOutputLayer: VerifierOutputLayer<F> + Serialize + for<'a> Deserialize<'a>;

    /// Returns the [LayerId] of the intermediate/input layer that this output
    /// layer is associated with.
    fn layer_id(&self) -> LayerId;

    /// Fix the variables of this output layer to random challenges sampled
    /// from the transcript.
    fn fix_layer(
        &mut self,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;

    /// Return the Circuit description for this Output Layer to be used
    /// by the verifier.
    fn into_verifier_output_layer(&self) -> Self::VerifierOutputLayer;
}

/// The interface of a Verifier's Output Layer.
pub trait VerifierOutputLayer<F: FieldExt> {
    /// Returns the [LayerId] of the intermediate/input layer that his output
    /// layer is associated with.
    fn layer_id(&self) -> LayerId;
}

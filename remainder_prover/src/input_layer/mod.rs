//! InputLayer and InputLayerDescription structs, as well as public- and ligero- input layers and
//! fiat-shamir challenges.
use crate::layer::LayerId;
use crate::mle::evals::MultilinearExtension;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

/// An input layer in order to generate random challenges for Fiat-Shamir.
pub mod fiat_shamir_challenge;
/// An input layer in which the input data is committed to using the Ligero PCS.
pub mod ligero_input_layer;

/// The prover's view of an input layer during circuit proving (undifferentiated as to type).
/// Note that, being undifferentiated, no functions for adding values or commitments to the transcript are provided.
#[derive(Debug, Clone)]
pub struct InputLayer<F: Field> {
    /// The MLE for the input layer.
    pub mle: MultilinearExtension<F>,
    /// The layer ID of the input layer.
    pub layer_id: LayerId,
}

impl<F: Field> InputLayer<F> {
    /// Create a new [InputLayer] from the given MLE, allocating the next available layer ID.
    pub fn new(mle: MultilinearExtension<F>) -> Self {
        let layer_id = LayerId::new_input_layer();
        Self { mle, layer_id }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash)]
/// The verifier's view of an input layer during circuit proving, containing
/// the shape information of this input layer.
pub struct InputLayerDescription {
    /// The layer ID of the input layer.
    pub layer_id: LayerId,
    /// The number of variables in the input layer.
    pub num_vars: usize,
}

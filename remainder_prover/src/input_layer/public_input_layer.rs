//! An input layer that is sent to the verifier in the clear
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{layer::LayerId, mle::evals::MultilinearExtension};

/// An Input Layer in which the data is sent to the verifier
/// "in the clear" (i.e. without a commitment).
#[derive(Debug, Clone)]
pub struct PublicInputLayer<F: Field> {
    /// The values of this input layer
    pub mle: MultilinearExtension<F>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash)]
/// The circuit description of a [PublicInputLayer] which stores
/// the shape information of this input layer.
pub struct PublicInputLayerDescription {
    /// The id of this input layer.
    pub layer_id: LayerId,
    /// The number of variables in this input layer, i.e. log2 of length.
    pub num_vars: usize,
}

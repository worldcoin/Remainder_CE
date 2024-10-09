//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use remainder_shared_types::{
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

/// An Input Layer in which the data is sent to the verifier
/// "in the clear" (i.e. without a commitment).
#[derive(Debug, Clone)]
pub struct PublicInputLayer<F: Field> {
    mle: MultilinearExtension<F>,
    pub(crate) layer_id: LayerId,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: Field")]
/// The circuit description of a [PublicInputLayer] which stores
/// the shape information of this input layer.
pub struct PublicInputLayerDescription<F: Field> {
    layer_id: LayerId,
    num_bits: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> PublicInputLayerDescription<F> {
    /// Constructor for the [PublicInputLayerDescription] using the layer_id
    /// and the number of variables in the MLE we are storing in the
    /// input layer.
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> PublicInputLayer<F> {
    /// FIXME(Ben) document or remove - could make self.mle public instead?
    pub fn get_evaluations_as_vec(&self) -> &Vec<F> {
        self.mle.get_evals_vector()
    }
}

impl<F: Field> PublicInputLayer<F> {
    /// Constructor for the [PublicInputLayer] using the MLE in the input
    /// and the layer_id.
    pub fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }
}
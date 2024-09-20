use std::marker::PhantomData;

use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::layer::LayerId;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: Field")]
/// The circuit description of a [HyraxInputLayer]. Stores the shape information of this layer.
/// All of the functionality of Hyrax input layers are taken care of in `remainder_hyrax/`, so
/// this is meant just to generate a circuit description.
pub struct CircuitHyraxInputLayer<F: Field> {
    /// The input layer ID.
    pub layer_id: LayerId,
    /// The number of variables this Hyrax Input Layer is on.
    num_bits: usize,
    /// The log number of columns in the matrix form of the data that
    /// will be committed to in this input layer.
    pub log_num_cols: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> CircuitHyraxInputLayer<F> {
    /// Constructor for the [CircuitHyraxInputLayer].
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        let log_num_cols = num_bits / 2;
        Self {
            layer_id,
            num_bits,
            log_num_cols,
            _marker: PhantomData,
        }
    }
}

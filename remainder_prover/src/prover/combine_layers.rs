//!Utilities for combining sub-circuits

use remainder_shared_types::{FieldExt, transcript::Transcript};

use crate::{layer::LayerId, mle::MleIndex};

use super::Layers;

///Utility for combining sub-circuits into a single circuit
/// DOES NOT WORK FOR GATE MLE
pub fn combine_layers<F: FieldExt, Tr: Transcript<F>>(layers: Vec<Layers<F, Tr>>) -> Layers<F, Tr> {
    todo!()
}

fn add_bits_to_layer_refs<F: FieldExt, Tr: Transcript<F>>(layers: &mut [Layers<F, Tr>], new_bits: Vec<MleIndex<F>>, effected_layer: LayerId) {
    todo!()
}
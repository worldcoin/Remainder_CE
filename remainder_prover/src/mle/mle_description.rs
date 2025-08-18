use remainder_shared_types::{extension_field::ExtensionField, transcript::VerifierTranscript, Field};
use serde::{Deserialize, Serialize};

use crate::{
    circuit_layout::CircuitEvalMap, expression::expr_errors::ExpressionError, layer::LayerId,
    mle::AbstractMle,
};

use super::{dense::DenseMle, verifier_mle::VerifierMle, MleIndex};

use anyhow::{anyhow, Result};

/// A metadata-only version of [crate::mle::dense::DenseMle] used in the Circuit
/// Descrption.  A [MleDescription] is stored in the leaves of an `Expression<F,
/// ExprDescription>` tree.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(bound = "F: Field")]
pub struct MleDescription<F: Field> {
    /// Layer whose data this MLE's is a subset of.
    layer_id: LayerId,

    /// A list of indices where the free variables have been assigned an index.
    var_indices: Vec<MleIndex<F>>,
}

impl<F: Field> AbstractMle<F> for MleDescription<F> {
    /// Returns the [LayerId] of this MleDescription.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Returns the MLE indices of this MleDescription.
    fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }
}

impl<F: Field> MleDescription<F> {
    /// Create a new [MleDescription] given its layer id and the [MleIndex]s that it holds.
    /// This is effectively the "shape" of a [DenseMle].
    pub fn new(layer_id: LayerId, var_indices: &[MleIndex<F>]) -> Self {
        Self {
            layer_id,
            var_indices: var_indices.to_vec(),
        }
    }

    /// Replace the current MLE indices stored with custom MLE indices. Most
    /// useful in [crate::layer::matmult::MatMult], where we do index manipulation.
    pub fn set_mle_indices(&mut self, new_mle_indices: Vec<MleIndex<F>>) {
        self.var_indices = new_mle_indices;
    }

    /// Convert [MleIndex::Free] into [MleIndex::Indexed] with the correct
    /// index labeling, given by start_index parameter.
    pub fn index_mle_indices(&mut self, start_index: usize) {
        let mut index_counter = start_index;
        self.var_indices
            .iter_mut()
            .for_each(|mle_index| match mle_index {
                MleIndex::Free => {
                    let indexed_mle_index = MleIndex::Indexed(index_counter);
                    index_counter += 1;
                    *mle_index = indexed_mle_index;
                }
                // If the MLE is indexed the same way, then skip
                // If the MLE is indexed a different way, panic
                // TODO (Benny): we need to handle the case that an MLE is indexed in two ways eventually
                //               perhaps through cloning?
                MleIndex::Indexed(i) => {
                    if *i == index_counter {
                        index_counter += 1;
                    } else {
                        panic!("Indexing the same MLE in two different ways is currently not supported!")
                    }
                }
                MleIndex::Fixed(_bit) => {}
                _ => panic!("We should not have indexed or bound bits at this point!"),
            });
    }

    /// Get the bits in the MLE that are fixed bits.
    pub fn prefix_bits(&self) -> Vec<bool> {
        self.var_indices
            .iter()
            .filter_map(|idx| match idx {
                MleIndex::Fixed(bit) => Some(*bit),
                _ => None,
            })
            .collect()
    }

    /// Convert this MLE into a [DenseMle] using the [CircuitMap],
    /// which holds information using the prefix bits and layer id
    /// on the data that should be stored in this MLE.
    pub fn into_dense_mle(&self, circuit_map: &CircuitEvalMap<F>) -> DenseMle<F> {
        let data = circuit_map.get_data_from_circuit_mle(self).unwrap();
        // DenseMle::new_with_prefix_bits((*data).clone(), self.layer_id(), self.prefix_bits())
        DenseMle::new_with_indices((*data).clone(), self.layer_id(), self.mle_indices())
    }

    /// Bind the variable with index `var_index` to `value`. Note that since
    /// [MleDescription] is the representation of a multilinear extension function
    /// sans data, it need not alter its internal MLE evaluations in any way.
    pub fn fix_variable(&mut self, var_index: usize, value: F) {
        for mle_index in self.var_indices.iter_mut() {
            if *mle_index == MleIndex::Indexed(var_index) {
                mle_index.bind_index(value);
            }
        }
    }

    /// Gets the values of the bound and fixed MLE indices of this MLE,
    /// panicking if the MLE is not fully bound.
    pub fn get_claim_point(&self, challenges: &[F]) -> Vec<F> {
        self.var_indices
            .iter()
            .map(|index| match index {
                MleIndex::Bound(chal, _idx) => *chal,
                MleIndex::Fixed(chal) => F::from(*chal as u64),
                MleIndex::Indexed(i) => challenges[*i],
                _ => panic!("DenseMleRefDesc contained free variables!"),
            })
            .collect()
    }
}

impl<E: ExtensionField> MleDescription<E> {
    /// Convert this MLE into a [VerifierMle], which represents a fully-bound MLE.
    pub fn into_verifier_mle(
        &self,
        point: &[E],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<VerifierMle<E>> {
        let verifier_indices = self
            .var_indices
            .iter()
            .map(|mle_index| match mle_index {
                MleIndex::Indexed(idx) => Ok(MleIndex::Bound(point[*idx], *idx)),
                MleIndex::Fixed(val) => Ok(MleIndex::Fixed(*val)),
                _ => Err(anyhow!(ExpressionError::EvaluateNotFullyIndexedError)),
            })
            .collect::<Result<Vec<MleIndex<E>>>>()?;

        let eval = transcript_reader.consume_extension_field_element("Fully bound MLE evaluation")?;

        Ok(VerifierMle::new(self.layer_id, verifier_indices, eval))
    }
}

use serde::{Deserialize, Serialize};
use shared_types::{transcript::VerifierTranscript, Field};

use crate::{
    circuit_layout::CircuitEvalMap, expression::expr_errors::ExpressionError, layer::LayerId,
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
                MleIndex::Fixed(_bit) => {}
                _ => panic!("We should not have indexed or bound bits at this point!"),
            });
    }

    /// Returns the [LayerId] of this MleDescription.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Returns the MLE indices of this MleDescription.
    pub fn var_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }

    /// The number of [MleIndex::Indexed] OR [MleIndex::Free] bits in this MLE.
    pub fn num_free_vars(&self) -> usize {
        self.var_indices.iter().fold(0, |acc, idx| {
            acc + match idx {
                MleIndex::Free => 1,
                MleIndex::Indexed(_) => 1,
                _ => 0,
            }
        })
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

    /// Convert this MLE into a [DenseMle] using the [CircuitEvalMap],
    /// which holds information using the prefix bits and layer id
    /// on the data that should be stored in this MLE.
    pub fn into_dense_mle(&self, circuit_map: &CircuitEvalMap<F>) -> DenseMle<F> {
        let data = circuit_map.get_data_from_circuit_mle(self).unwrap();
        DenseMle::new_with_prefix_bits((*data).clone(), self.layer_id(), self.prefix_bits())
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

    /// Convert this MLE into a [VerifierMle], which represents a fully-bound MLE.
    pub fn into_verifier_mle(
        &self,
        point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierMle<F>> {
        let verifier_indices = self
            .var_indices
            .iter()
            .map(|mle_index| match mle_index {
                MleIndex::Indexed(idx) => Ok(MleIndex::Bound(point[*idx], *idx)),
                MleIndex::Fixed(val) => Ok(MleIndex::Fixed(*val)),
                _ => Err(anyhow!(ExpressionError::SelectorBitNotBoundError)),
            })
            .collect::<Result<Vec<MleIndex<F>>>>()?;

        let eval = transcript_reader.consume_element("Fully bound MLE evaluation")?;

        Ok(VerifierMle::new(self.layer_id, verifier_indices, eval))
    }
}

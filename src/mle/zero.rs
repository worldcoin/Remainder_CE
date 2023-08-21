//!Contains ZeroMleRef which is an MleRef which always contains only all zeros

//!Contains ZeroMleRef which is an MleRef which always contains only all zeros

use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};

use crate::layer::{Claim, LayerId};
use lcpc_2d::FieldExt;

use super::{MleIndex, MleRef, mle_enum::MleEnum};

///An MLERef that is only zeros; Typically used for the output layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMleRef<F: FieldExt> {
    mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck)
    num_vars: usize,
    layer_id: LayerId,
    zero: [F; 1],
    indexed: bool,
}

impl<F: FieldExt> ZeroMleRef<F> {
    ///Creates a new ZERO MleRef
    pub fn new(num_vars: usize, prefix_bits: Option<Vec<MleIndex<F>>>, layer_id: LayerId) -> Self {
        let mle_indices = prefix_bits
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Iterated, num_vars))
            .collect_vec();

        Self {
            mle_indices,
            num_vars,
            layer_id,
            zero: [F::zero()],
            indexed: false,
        }
    }
}

impl<F: FieldExt> MleRef for ZeroMleRef<F> {
    type F = F;

    fn bookkeeping_table(&self) -> &[Self::F] {
        &self.zero
    }

    fn indexed(&self) -> bool {
        self.indexed
    }

    fn mle_indices(&self) -> &[MleIndex<Self::F>] {
        &self.mle_indices
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::IndexedBit(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        // --- One fewer iterated bit to sumcheck through ---
        self.num_vars -= 1;

        if self.num_vars == 0 {
            Some((
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                F::zero(),
            ))
        } else {
            None
        }
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Iterated {
                *mle_index = MleIndex::IndexedBit(curr_index + new_indices);
                new_indices += 1;
            }
        }

        curr_index + new_indices
    }

    fn get_layer_id(&self) -> LayerId {
        self.layer_id.clone()
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        MleEnum::Zero(self)
    }
}

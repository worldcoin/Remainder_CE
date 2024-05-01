//! A space-efficient implementation of an [MleRef] which contains only zeros.

use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};

use crate::claims::{wlx_eval::ClaimMle, Claim};
use crate::claims::{ClaimError, YieldClaim};
use crate::layer::{LayerError, LayerId};
use remainder_shared_types::FieldExt;

use super::{mle_enum::MleEnum, MleIndex, MleRef};

/// An [MleRef] that contains only zeros; typically used for the output layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroMleRef<F> {
    pub(crate) mle_indices: Vec<MleIndex<F>>,
    pub(crate) original_mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck).
    num_vars: usize,
    pub(crate) layer_id: LayerId,
    zero: [F; 1],
    indexed: bool,
}

impl<F: FieldExt> ZeroMleRef<F> {
    /// Constructs a new `ZeroMleRef` on `num_vars` variables with the
    /// appropriate `prefix_bits` for a layer with ID `layer_id`.
    pub fn new(num_vars: usize, prefix_bits: Option<Vec<MleIndex<F>>>, layer_id: LayerId) -> Self {
        let mle_indices = prefix_bits
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Iterated, num_vars))
            .collect_vec();

        Self {
            mle_indices: mle_indices.clone(),
            original_mle_indices: mle_indices,
            num_vars,
            layer_id,
            zero: [F::ZERO],
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

    fn original_mle_indices(&self) -> &Vec<MleIndex<Self::F>> {
        &self.original_mle_indices
    }

    fn original_bookkeeping_table(&self) -> &[Self::F] {
        &self.zero
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn original_num_vars(&self) -> usize {
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
            let send_claim = Claim::new(
                self.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                F::ZERO,
            );
            Some(send_claim)
        } else {
            None
        }
    }

    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<Claim<Self::F>> {
        self.fix_variable(indexed_bit_index, point)
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
        self.layer_id
    }

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]) {
        self.mle_indices.append(&mut new_indices.to_vec());
    }

    fn get_enum(self) -> MleEnum<Self::F> {
        MleEnum::Zero(self)
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for ZeroMleRef<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        if self.bookkeeping_table().len() != 1 {
            return Err(LayerError::ClaimError(ClaimError::MleRefMleError));
        }
        let mle_indices: Result<Vec<F>, _> = self
            .mle_indices
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::MleRefMleError))
            })
            .collect();
        Ok(vec![ClaimMle::new(
            mle_indices?,
            F::ZERO,
            None,
            Some(self.layer_id),
            Some(self.clone().get_enum()),
        )])
    }
}

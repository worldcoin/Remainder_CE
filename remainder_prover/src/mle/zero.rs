//! A space-efficient implementation of an [MleRef] which contains only zeros.

use itertools::{repeat_n, Itertools};
use serde::{Deserialize, Serialize};

use crate::claims::{wlx_eval::ClaimMle, Claim};
use crate::claims::{ClaimError, YieldClaim};
use crate::layer::{LayerError, LayerId};
use remainder_shared_types::Field;

use super::Mle;
use super::{mle_enum::MleEnum, MleIndex};

/// An [MleRef] that contains only zeros; typically used for the output layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZeroMle<F> {
    pub(crate) mle_indices: Vec<MleIndex<F>>,
    pub(crate) original_mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck).
    pub(crate) num_vars: usize,
    pub(crate) layer_id: LayerId,
    pub(crate) zero: [F; 1],
    pub(crate) indexed: bool,
}

impl<F: Field> ZeroMle<F> {
    /// Constructs a new `ZeroMle` on `num_vars` variables with the
    /// appropriate `prefix_bits` for a layer with ID `layer_id`.
    pub fn new(num_vars: usize, prefix_bits: Option<Vec<MleIndex<F>>>, layer_id: LayerId) -> Self {
        let mle_indices = prefix_bits
            .into_iter()
            .flatten()
            .chain(repeat_n(MleIndex::Free, num_vars))
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

impl<F: Field> Mle<F> for ZeroMle<F> {
    fn bookkeeping_table(&self) -> &[F] {
        &self.zero
    }

    fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.mle_indices
    }

    fn original_mle_indices(&self) -> &Vec<MleIndex<F>> {
        &self.original_mle_indices
    }

    fn original_bookkeeping_table(&self) -> &[F] {
        &self.zero
    }

    fn num_free_vars(&self) -> usize {
        self.num_vars
    }

    fn original_num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_variable(&mut self, round_index: usize, challenge: F) -> Option<Claim<F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Indexed(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        // --- One fewer free variable to sumcheck through ---
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

    fn fix_variable_at_index(&mut self, indexed_bit_index: usize, point: F) -> Option<Claim<F>> {
        self.fix_variable(indexed_bit_index, point)
    }

    fn index_mle_indices(&mut self, curr_index: usize) -> usize {
        let mut new_indices = 0;
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Free {
                *mle_index = MleIndex::Indexed(curr_index + new_indices);
                new_indices += 1;
            }
        }

        curr_index + new_indices
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_enum(self) -> MleEnum<F> {
        MleEnum::Zero(self)
    }

    #[doc = " Get the padded set of evaluations over the boolean hypercube; useful for"]
    #[doc = " constructing the input layer."]
    fn get_padded_evaluations(&self) -> Vec<F> {
        todo!()
    }

    #[doc = " Mutates the MLE in order to set the prefix bits. This is needed when we"]
    #[doc = " are working with dataparallel circuits and new bits need to be added."]
    fn add_prefix_bits(&mut self, _new_bits: Vec<MleIndex<F>>) {
        todo!()
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for ZeroMle<F> {
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

        // Note: Claim result is always zero. No need to append to transcript.

        Ok(vec![ClaimMle::new(
            mle_indices?,
            F::ZERO,
            None,
            Some(self.layer_id),
            Some(self.clone().get_enum()),
        )])
    }
}

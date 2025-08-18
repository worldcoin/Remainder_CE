//! A space-efficient implementation of an MLE which contains only zeros.

use itertools::{repeat_n, Itertools};
use remainder_shared_types::field::ExtensionField;
use serde::{Deserialize, Serialize};

use crate::claims::RawClaim;
use crate::layer::LayerId;
use crate::mle::mle_enum::LiftTo;
use crate::mle::AbstractMle;
use remainder_shared_types::Field;

use super::evals::{Evaluations, EvaluationsIterator};
use super::Mle;
use super::{mle_enum::MleEnum, MleIndex};

/// An MLE that contains only zeros; typically used for the output layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(bound = "F: Field")]
pub struct ZeroMle<F: Field> {
    pub(crate) mle_indices: Vec<MleIndex<F>>,
    /// Number of non-fixed variables within this MLE
    /// (warning: this gets modified destructively DURING sumcheck).
    pub(crate) num_vars: usize,
    pub(crate) layer_id: LayerId,
    pub(crate) zero: [F; 1],
    pub(crate) zero_eval: Evaluations<F>,
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
            mle_indices,
            num_vars,
            layer_id,
            zero: [F::ZERO],
            zero_eval: Evaluations::new(num_vars, vec![F::ZERO]),
            indexed: false,
        }
    }
}

impl<F: Field> AbstractMle<F> for ZeroMle<F> {
    fn mle_indices(&self) -> &[MleIndex<F>] {
        &self.mle_indices
    }

    fn num_free_vars(&self) -> usize {
        self.num_vars
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}
impl<F: Field> Mle<F> for ZeroMle<F> {
    type ExtendedMle<E: ExtensionField<F>> = ZeroMle<E>;

    fn fix_variable(&mut self, round_index: usize, challenge: F) -> Option<RawClaim<F>> {
        for mle_index in self.mle_indices.iter_mut() {
            if *mle_index == MleIndex::Indexed(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        // One fewer free variable to sumcheck through
        self.num_vars -= 1;

        if self.num_vars == 0 {
            let send_claim = RawClaim::new(
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

    fn fix_variable_at_index(&mut self, indexed_bit_index: usize, point: F) -> Option<RawClaim<F>> {
        self.fix_variable(indexed_bit_index, point)
    }

    fn fix_variable_ext<E: ExtensionField<F>>(mle: &Self, round_index: usize, challenge: E) -> (
        ZeroMle<E>,
        Option<RawClaim<E>>,
     ) {
        let mut new_mle_indices = mle.mle_indices.lift();

        for mle_index in new_mle_indices.iter_mut() {
            if *mle_index == MleIndex::Indexed(round_index) {
                mle_index.bind_index(challenge);
            }
        }

        // One fewer free variable to sumcheck through
        let new_mle = ZeroMle::<E> {
            mle_indices: new_mle_indices,
            num_vars: mle.num_vars - 1,
            layer_id: mle.layer_id,
            zero: [mle.zero[0].into()],
            zero_eval: mle.zero_eval.lift(),
            indexed: mle.indexed,
        };

        if new_mle.num_vars == 0 {
            let send_claim = RawClaim::new(
                new_mle.mle_indices
                    .iter()
                    .map(|index| index.val().unwrap())
                    .collect_vec(),
                E::ZERO,
            );
            (new_mle, Some(send_claim))
        } else {
            (new_mle, None)
        }
    }

    fn fix_variable_at_index_ext<E: ExtensionField<F>>(mle: &Self, indexed_bit_index: usize, point: E) -> (
        ZeroMle<E>,
        Option<RawClaim<E>>,
     ) {
        Self::fix_variable_ext(mle, indexed_bit_index, point)
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

    fn len(&self) -> usize {
        1
    }

    fn iter(&self) -> EvaluationsIterator<F> {
        self.zero_eval.iter()
    }

    fn first(&self) -> F {
        F::ZERO
    }

    fn value(&self) -> F {
        assert!(self.is_fully_bounded());
        F::ZERO
    }

    fn get(&self, index: usize) -> Option<F> {
        if index < self.len() {
            Some(F::ZERO)
        } else {
            None
        }
    }
}

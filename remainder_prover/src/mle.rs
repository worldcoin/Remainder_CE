//! An [Mle] is a Multilinear Extension that contains a more complex type (i.e.
//! `T`, or `(T, T)` or `ExampleStruct`)

use core::fmt::Debug;

use evals::EvaluationsIterator;
use serde::{Deserialize, Serialize};

use crate::{claims::RawClaim, layer::LayerId};
use remainder_shared_types::Field;

use self::mle_enum::MleEnum;

/// Contains the default, dense implementation of an [Mle].
pub mod dense;

/// Contains an implementation of the zero [Mle].
pub mod zero;

/// Contains a wrapper type around [Mle] implementations.
pub mod mle_enum;

/// Defines [betavalues::BetaValues], a struct for storing the bookkeeping
/// tables of Beta functions.
pub mod betavalues;

/// Defines [crate::mle::evals::Evaluations] and
/// [crate::mle::evals::MultilinearExtension] for representing un-indexed MLEs.
pub mod evals;

/// Defines [crate::mle::mle_description::MleDescription], i.e. the in-circuit-context description
/// of a [crate::mle::evals::MultilinearExtension] which includes "prefix vars" and "free vars" but
/// not the actual evaluations of the function over the hypercube.
pub mod mle_description;

/// Defines [crate::mle::verifier_mle::VerifierMle], i.e. the verifier's view of a "fully-bound" MLE
/// with a prover-claimed value.
pub mod verifier_mle;

/// Abstract structure of an Mle
/// Including its number of variables, indices,layer_id, but not evaluations
pub trait AbstractMle: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// Returns the number of free variables this Mle is defined on.
    /// Equivalently, this is the log_2 of the size of the unpruned bookkeeping
    /// table.
    /// The number of [MleIndex::Indexed] OR [MleIndex::Free] bits in this MLE.
    fn num_free_vars(&self) -> usize {
        self.mle_indices().iter().fold(0, |acc, idx| {
            acc + match idx {
                MleIndex::Free => 1,
                MleIndex::Indexed(_) => 1,
                _ => 0,
            }
        })
    }

    /// AN MLE is unbounded if every variable is Fixed or Indexed
    fn is_unbounded(&self) -> bool {
        self.mle_indices().iter().fold(true, |acc, idx| {
            acc && match idx {
                MleIndex::Free | MleIndex::Fixed(_) | MleIndex::Indexed(_) => true,
                _ => false,
            }
        })
    }

    /// obtain the length of the bind list
    fn get_bind_list_len(&self) -> usize {
        if let Some(i) = self
            .mle_indices()
            .iter()
            .filter_map(|idx| match idx {
                MleIndex::Indexed(i) => Some(*i),
                _ => None,
            })
            .max()
        {
            i + 1
        } else {
            0
        }
    }

    /// Returns an empty bind list
    fn init_bind_list<F: Field>(&self) -> Vec<Option<F>> {
        assert!(self.is_unbounded());
        vec![None; self.get_bind_list_len()]
    }

    /// An MLE is fully bounded if it has no more free variables
    /// and there is a corresponding list of bound values.
    fn is_fully_bounded<T>(&self, bind_list: &Vec<Option<T>>) -> bool {
        self.mle_indices().iter().fold(true, |acc, idx| {
            acc && match idx {
                MleIndex::Fixed(_) => true,
                MleIndex::Bound(idx) => bind_list[*idx].is_some(),
                _ => false,
            }
        })
    }

    /// Get the indicies of the `Mle` that this `MleRef` represents.
    fn mle_indices(&self) -> &[MleIndex];

    /// Get the layer ID of the associated MLE.
    fn layer_id(&self) -> LayerId;
}

// TODO!(Maybe this type needs PartialEq, could be easily implemented with a
// random id...).
/// The trait that defines how a semantic Type (T) and a MultiLinearEvaluation
/// containing field elements (F) interact. T should always be a composite type
/// containing Fs. For example (F, F) or a struct containing Fs.
///
/// If you want to construct an Mle, or use an Mle for some non-cryptographic
/// computation (e.g. wit gen) then you should always use the iterator adaptors
/// IntoIterator and FromIterator, this is to ensure that the semantic ordering
/// within T is always consistent.
#[allow(clippy::len_without_is_empty)]
pub trait Mle<F: Field>: Clone + Debug + Send + Sync + AbstractMle {
    /// Get the padded set of evaluations over the boolean hypercube; useful for
    /// constructing the input layer.
    fn get_padded_evaluations(&self) -> Vec<F>;

    /// Mutates the MLE in order to set the prefix bits. This is needed when we
    /// are working with dataparallel circuits and new bits need to be added.
    fn add_prefix_bits(&mut self, new_bits: Vec<MleIndex>);

    /// Returns the length of the current bookkeeping table.
    fn len(&self) -> usize;

    /// Returns an iterator over the evaluations of the current MLE.
    fn iter(&self) -> EvaluationsIterator<F>;

    /// Returns the first element in the bookkeeping table corresponding to the
    /// value of this Dense MLE when all free variables are set to zero. This
    /// operations never panics (see [evals::MultilinearExtension::first])
    fn first(&self) -> F;

    /// If this is a fully-bound Dense MLE, it returns its value.
    /// Otherwise panics.
    fn value(&self) -> F;

    /// Returns the first element of the evaluations table (if any).
    fn get(&self, index: usize) -> Option<F>;

    /// Fix the variable at `round_index` at a given `challenge` point. Mutates
    /// `self` to be the bookeeping table for the new MLE.
    ///
    /// If the new MLE becomes fully bound, returns the evaluation of the fully
    /// bound Mle.
    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: F,
        bind_list: &mut Vec<Option<F>>,
    ) -> Option<RawClaim<F>>;

    /// Fix the (indexed) free variable at `indexed_bit_index` with a given
    /// challenge `point`. Mutates `self`` to be the bookeeping table for the
    /// new MLE.  If the new MLE becomes fully bound, returns the evaluation of
    /// the fully bound MLE in the form of a [crate::claims::RawClaim].
    ///
    /// # Panics
    /// If `indexed_bit_index` does not correspond to a
    /// `MleIndex::Indexed(indexed_bit_index)` in `mle_indices`.
    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: F,
        bind_list: &mut Vec<Option<F>>,
    ) -> Option<RawClaim<F>>;

    /// Similar to `fix_variable_at_index`, but do not keep track of the bound point
    /// Useful for only obtaining the final evaluation value
    fn fix_variable_at_index_no_bind_list(&mut self, indexed_bit_index: usize, point: F);

    /// Mutates the [MleIndex]es stored in `self` that are [MleIndex::Free] and
    /// turns them into [MleIndex::Indexed] with the bit index being determined
    /// from `curr_index`.
    /// Returns the `(curr_index + number of IndexedBits now in the MleIndices)`.
    fn index_mle_indices(&mut self, curr_index: usize, bind_list: &mut Vec<Option<F>>) -> usize;

    /// Similar to `index_mle_indices``, but without modifying a bind list
    fn index_mle_indices_no_bind_list(&mut self, curr_index: usize) -> usize;

    /// Get the associated enum that this MLE is a part of ([MleEnum::Dense] or [MleEnum::Zero]).
    fn get_enum(self) -> MleEnum<F>;
}

/// Represents all the possible types of indices for an [Mle].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum MleIndex {
    /// A "selector" bit for fixed MLE access.
    Fixed(bool),

    /// An "unbound" bit that iterates over the contents of the MLE.
    Free,

    /// An "unbound" bit where the particular b_i in the larger expression has
    /// been set.
    Indexed(usize),

    /// An index that has been bound to a random challenge by the sumcheck
    /// protocol. The random challenge is obtained in a separate list.
    Bound(usize),
}

impl MleIndex {
    /// If `self` is a [MleIndex::Free] bit, turns it into an
    /// [MleIndex::Indexed] with index `bit`.  Otherwise, `self` is not
    /// modified.
    pub fn index_var(&mut self, bit: usize) {
        if let MleIndex::Free = self {
            *self = Self::Indexed(bit)
        }
    }

    /// If `self` is `Indexed(idx)`, bind it to `chal`, i.e. turn it into
    /// a `Bound(chal, idx)` variant. Otherwise, `self` is not modified.
    pub fn bind_index<F: Field>(&mut self, chal: F, bind_list: &mut Vec<Option<F>>) {
        if let MleIndex::Indexed(bit) = self {
            if let Some(old_chal) = bind_list[*bit] {
                if old_chal != chal {
                    panic!("Conflicting bind value to index {bit}: {old_chal:?} and {chal:?}")
                }
            }
            bind_list[*bit] = Some(chal);
            *self = Self::Bound(*bit);
        }
    }

    /// Similar to `bind_index`, but do not keep track of bind_list
    pub fn bind_index_no_check(&mut self) {
        if let MleIndex::Indexed(bit) = self {
            *self = Self::Bound(*bit);
        }
    }

    /// Evaluate this `MleIndex`.
    /// * `Fixed(bit)` evaluates to `0` or `1` when `bit` is `false` or `true`
    ///   respectively.
    /// * `Free` bits are *not* evaluated, i.e. `None` is returned.
    /// * `IndexedBit(idx)` bits are *not* evaluated, i.e. `None` is returned.
    /// * `Bound(chal, idx)` variants evaluate to `chal`.
    pub fn val<F: Field>(&self, bind_list: &Vec<Option<F>>) -> Option<F> {
        match self {
            MleIndex::Fixed(true) => Some(F::ONE),
            MleIndex::Fixed(false) => Some(F::ZERO),
            MleIndex::Bound(bit) => Some(bind_list[*bit].unwrap()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::MleIndex;
    use remainder_shared_types::Fr;

    #[test]
    fn test_mle_index_val() {
        let mut bind_list: Vec<Option<Fr>> = vec![None; 100];

        // 0 selector bit.
        assert_eq!(MleIndex::Fixed(false).val(&bind_list), Some(Fr::zero()));

        // 1 selector bit.
        assert_eq!(MleIndex::Fixed(true).val(&bind_list), Some(Fr::one()));

        // Free index.
        assert_eq!(MleIndex::Free.val(&bind_list), None);

        // Indexed bit index.
        assert_eq!(MleIndex::Indexed(42).val(&bind_list), None);

        // Bound index.
        bind_list[0] = Some(Fr::from(42));
        assert_eq!(MleIndex::Bound(0).val(&bind_list), Some(Fr::from(42)));
    }

    #[test]
    fn test_mle_index_bind() {
        // An `Fixed` index remains unaffected.
        let mut mle_index: MleIndex = MleIndex::Fixed(true);
        let mut bind_list: Vec<Option<Fr>> = Vec::new();
        mle_index.bind_index(Fr::from(17), &mut bind_list);
        assert_eq!(mle_index, MleIndex::Fixed(true));

        // A `Free` index remains unaffected.
        let mut mle_index: MleIndex = MleIndex::Free;
        let mut bind_list: Vec<Option<Fr>> = Vec::new();
        mle_index.bind_index(Fr::from(17), &mut bind_list);
        assert_eq!(mle_index, MleIndex::Free);

        // An `IndexedBit` index gets bound.
        let mut mle_index: MleIndex = MleIndex::Indexed(42);
        let mut bind_list: Vec<Option<Fr>> = vec![None; 100];
        mle_index.bind_index(Fr::from(17), &mut bind_list);
        assert_eq!(mle_index, MleIndex::Bound(42));
        assert_eq!(mle_index.val(&bind_list), Some(Fr::from(17)));

        // An `Bound` index remains unaffected.
        mle_index.bind_index(Fr::from(37), &mut bind_list);
        assert_eq!(mle_index, MleIndex::Bound(42));
        assert_eq!(mle_index.val(&bind_list), Some(Fr::from(17)));
    }

    #[test]
    fn test_mle_index_index() {
        // An `Fixed` index remains unaffected.
        let mut mle_index: MleIndex = MleIndex::Fixed(true);
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::Fixed(true));

        // A `Free` index becomes `IndexedBit`.
        let mut mle_index: MleIndex = MleIndex::Free;
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::Indexed(17));

        // An `IndexedBit` index remains unaffected.
        let mut mle_index: MleIndex = MleIndex::Indexed(42);
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::Indexed(42));

        // An `Bound` index remains unaffected.
        let mut mle_index: MleIndex = MleIndex::Indexed(42);
        let mut bind_list: Vec<Option<Fr>> = vec![None; 100];
        mle_index.bind_index(Fr::from(17), &mut bind_list);
        mle_index.index_var(37);
        assert_eq!(mle_index, MleIndex::Bound(42));
        assert_eq!(mle_index.val(&bind_list), Some(Fr::from(17)));
    }
}

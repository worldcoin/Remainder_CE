//! An [Mle] is a Multilinear Extension that contains a more complex type (i.e.
//! `T`, or `(T, T)` or `ExampleStruct`)

use core::fmt::Debug;

use evals::EvaluationsIterator;
use serde::{Deserialize, Serialize};

use crate::{claims::RawClaim, layer::LayerId, mle::mle_enum::LiftTo};
use remainder_shared_types::{extension_field::ExtensionField, Field};

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
pub trait AbstractMle<F: Field>:
    Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
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

    /// An MLE is fully bounded if it has no more free variables.
    fn is_fully_bounded(&self) -> bool {
        self.num_free_vars() == 0
    }

    /// Get the indicies of the `Mle` that this `MleRef` represents.
    fn mle_indices(&self) -> &[MleIndex<F>];

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
pub trait Mle<F: Field>: Clone + Debug + Send + Sync + AbstractMle<F> {
    /// Get the padded set of evaluations over the boolean hypercube; useful for
    /// constructing the input layer.
    fn get_padded_evaluations(&self) -> Vec<F>;

    /// Mutates the MLE in order to set the prefix bits. This is needed when we
    /// are working with dataparallel circuits and new bits need to be added.
    fn add_prefix_bits(&mut self, new_bits: Vec<MleIndex<F>>);

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
    fn fix_variable(&mut self, round_index: usize, challenge: F) -> Option<RawClaim<F>>;

    /// Fix the (indexed) free variable at `indexed_bit_index` with a given
    /// challenge `point`. Mutates `self`` to be the bookeeping table for the
    /// new MLE.  If the new MLE becomes fully bound, returns the evaluation of
    /// the fully bound MLE in the form of a [crate::claims::RawClaim].
    ///
    /// # Panics
    /// If `indexed_bit_index` does not correspond to a
    /// `MleIndex::Indexed(indexed_bit_index)` in `mle_indices`.
    fn fix_variable_at_index(&mut self, indexed_bit_index: usize, point: F) -> Option<RawClaim<F>>;

    /// Mutates the [MleIndex]es stored in `self` that are [MleIndex::Free] and
    /// turns them into [MleIndex::Indexed] with the bit index being determined
    /// from `curr_index`.
    /// Returns the `(curr_index + number of IndexedBits now in the MleIndices)`.
    fn index_mle_indices(&mut self, curr_index: usize) -> usize;

    /// Get the associated enum that this MLE is a part of ([MleEnum::Dense] or [MleEnum::Zero]).
    fn get_enum(self) -> MleEnum<F>;
}

/// Represents all the possible types of indices for an [Mle].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum MleIndex<F> {
    /// A "selector" bit for fixed MLE access.
    Fixed(bool),

    /// An "unbound" bit that iterates over the contents of the MLE.
    Free,

    /// An "unbound" bit where the particular b_i in the larger expression has
    /// been set.
    Indexed(usize),

    /// An index that has been bound to a random challenge by the sumcheck
    /// protocol.
    Bound(F, usize),
}

impl<F: Field> MleIndex<F> {
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
    pub fn bind_index(&mut self, chal: F) {
        if let MleIndex::Indexed(bit) = self {
            *self = Self::Bound(chal, *bit)
        }
    }

    /// Evaluate this `MleIndex`.
    /// * `Fixed(bit)` evaluates to `0` or `1` when `bit` is `false` or `true`
    ///   respectively.
    /// * `Free` bits are *not* evaluated, i.e. `None` is returned.
    /// * `IndexedBit(idx)` bits are *not* evaluated, i.e. `None` is returned.
    /// * `Bound(chal, idx)` variants evaluate to `chal`.
    pub fn val(&self) -> Option<F> {
        match self {
            MleIndex::Fixed(true) => Some(F::ONE),
            MleIndex::Fixed(false) => Some(F::ZERO),
            MleIndex::Bound(chal, _) => Some(*chal),
            _ => None,
        }
    }
}

/// Simple lift from [MleIndex<F>] to [MleIndex<E>].
impl<E: ExtensionField> LiftTo<MleIndex<E>> for MleIndex<E::BaseField> {
    fn lift(self) -> MleIndex<E> {
        match self {
            MleIndex::Fixed(val) => MleIndex::Fixed(val),
            MleIndex::Free => MleIndex::Free,
            MleIndex::Indexed(var_idx) => MleIndex::Indexed(var_idx),
            MleIndex::Bound(base_chal, var_idx) => MleIndex::Bound(base_chal.into(), var_idx),
        }
    }
}

/// Simple lift from [Vec<MleIndex<F>>] to [Vec<MleIndex<E>>].
impl<E: ExtensionField> LiftTo<Vec<MleIndex<E>>> for Vec<MleIndex<E::BaseField>> {
    fn lift(self) -> Vec<MleIndex<E>> {
        self.iter().map(|i| i.clone().lift()).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod test {
    use super::MleIndex;
    use remainder_shared_types::Fr;

    #[test]
    fn test_mle_index_val() {
        // 0 selector bit.
        assert_eq!(MleIndex::<Fr>::Fixed(false).val(), Some(Fr::zero()));

        // 1 selector bit.
        assert_eq!(MleIndex::<Fr>::Fixed(true).val(), Some(Fr::one()));

        // Bound index.
        assert_eq!(
            MleIndex::<Fr>::Bound(Fr::from(42), 0).val(),
            Some(Fr::from(42))
        );

        // Free index.
        assert_eq!(MleIndex::<Fr>::Free.val(), None);

        // Indexed bit index.
        assert_eq!(MleIndex::<Fr>::Indexed(42).val(), None);
    }

    #[test]
    fn test_mle_index_bind() {
        // An `Fixed` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Fixed(true);
        mle_index.bind_index(Fr::from(17));
        assert_eq!(mle_index, MleIndex::<Fr>::Fixed(true));

        // A `Free` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Free;
        mle_index.bind_index(Fr::from(17));
        assert_eq!(mle_index, MleIndex::<Fr>::Free);

        // An `IndexedBit` index gets bound.
        let mut mle_index: MleIndex<Fr> = MleIndex::Indexed(42);
        mle_index.bind_index(Fr::from(17));
        assert_eq!(mle_index, MleIndex::<Fr>::Bound(Fr::from(17), 42));

        // An `Bound` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Bound(Fr::from(17), 42);
        mle_index.bind_index(Fr::from(37));
        assert_eq!(mle_index, MleIndex::<Fr>::Bound(Fr::from(17), 42));
    }

    #[test]
    fn test_mle_index_index() {
        // An `Fixed` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Fixed(true);
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::<Fr>::Fixed(true));

        // A `Free` index becomes `IndexedBit`.
        let mut mle_index: MleIndex<Fr> = MleIndex::Free;
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::<Fr>::Indexed(17));

        // An `IndexedBit` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Indexed(42);
        mle_index.index_var(17);
        assert_eq!(mle_index, MleIndex::<Fr>::Indexed(42));

        // An `Bound` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Bound(Fr::from(17), 42);
        mle_index.index_var(37);
        assert_eq!(mle_index, MleIndex::<Fr>::Bound(Fr::from(17), 42));
    }
}

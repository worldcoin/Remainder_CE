//! An [Mle] is a Multilinear Extension that contains a more complex type (i.e.
//! `T`, or `(T, T)` or `ExampleStruct`)

use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::{claims::RawClaim, layer::LayerId};
use remainder_shared_types::Field;

use self::mle_enum::MleEnum;
use dyn_clonable::*;

/// Contains the default, dense implementation of an [Mle].
pub mod dense;

/// Contains an implementation of the zero [Mle].
pub mod zero;

/// Contains a wrapper type around [Mle] implementations.
pub mod mle_enum;

/// Defines [betavalues::BetaValues], a struct for storing the bookkeeping
/// tables of Beta functions.
pub mod betavalues;

/// Defines [Evaluations], [MultilinearExtension] for representing un-indexed
/// MLEs.
pub mod evals;

/// Defines trait/struct relevant for Mles (as input) in circuit.
pub mod bundled_input_mle;

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
#[clonable]
pub trait Mle<F: Field>: Clone + Debug + Send + Sync {
    /// Returns the number of free variables this Mle is defined on.
    /// Equivalently, this is the log_2 of the size of the *whole* bookkeeping
    /// table.
    fn num_free_vars(&self) -> usize;

    /// Get the padded set of evaluations over the boolean hypercube; useful for
    /// constructing the input layer.
    fn get_padded_evaluations(&self) -> Vec<F>;

    /// Mutates the MLE in order to set the prefix bits. This is needed when we
    /// are working with dataparallel circuits and new bits need to be added.
    fn add_prefix_bits(&mut self, new_bits: Vec<MleIndex<F>>);

    /// Get the layer ID of the associated MLE.
    fn layer_id(&self) -> LayerId;

    /// below are methods that belonged to MleRef originally

    /// Gets reference to the current bookkeeping tables.
    fn bookkeeping_table(&self) -> &[F];

    /// Get the indicies of the `Mle` that this `MleRef` represents.
    fn mle_indices(&self) -> &[MleIndex<F>];

    /// Fix the variable at `round_index` at a given `challenge` point. Mutates
    /// `self` to be the bookeeping table for the new MLE.
    ///
    /// If the new MLE becomes fully bound, returns the evaluation of the fully
    /// bound Mle.
    fn fix_variable(&mut self, round_index: usize, challenge: F) -> Option<RawClaim<F>>;

    /// Fix the (indexed) free variable at `indexed_bit_index` with a given
    /// challenge `point`. Mutates `self`` to be the bookeeping table for the
    /// new MLE.  If the new MLE becomes fully bound, returns the evaluation of
    /// the fully bound MLE in the form of a [Claim].
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

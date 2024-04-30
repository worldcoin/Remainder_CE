//! An [Mle] is a Multilinear Extension that contains a more complex type (i.e.
//! `T`, or `(T, T)` or `ExampleStruct`)

use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::claims::Claim;
use crate::layer::LayerId;
use remainder_shared_types::FieldExt;

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
pub trait Mle<F: FieldExt>: Clone {
    /// Returns the number of iterated variables this Mle is defined on.
    /// Equivalently, this is the log_2 of the size of the *whole* bookkeeping
    /// table.
    fn num_iterated_vars(&self) -> usize;

    /// Get the padded set of evaluations over the boolean hypercube; useful for
    /// constructing the input layer.
    fn get_padded_evaluations(&self) -> Vec<F>;

    /// Mutates the MLE in order to set the prefix bits. This is needed when we
    /// are working with dataparallel circuits and new bits need to be added.
    fn set_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>);

    /// Gets the prefix bits currently stored in the MLE. This is needed when
    /// prefix bits are generated after combining MLEs.
    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>>;

    /// Get the layer ID of the associated MLE.
    fn layer_id(&self) -> LayerId;
}

/// `MleRef` keeps track of an [Mle] and the fixed indices of the `Mle` to be
/// used in an expression.
pub trait MleRef: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    /// The Field Element this MleRef refers to.
    type F: FieldExt;

    /// Gets reference to the current bookkeeping tables.
    fn bookkeeping_table(&self) -> &[Self::F];

    /// Get the indicies of the `Mle` that this `MleRef` represents.
    fn mle_indices(&self) -> &[MleIndex<Self::F>];

    /// Gets the original, unmutated MLE indices associated with an MLE
    /// when it was first created (before any variable binding occured).
    fn original_mle_indices(&self) -> &Vec<MleIndex<Self::F>>;

    /// Gets the original, unmutated MLE bookkeeping table associated with an MLE
    /// when it was first created (before any variable binding occured).
    fn original_bookkeeping_table(&self) -> &[Self::F];

    /// Add new indices at the end of an MLE.
    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]);

    /// Number of variables the [Mle] this is a reference to is over.
    fn num_vars(&self) -> usize;

    /// Number of original variables, not mutated.
    fn original_num_vars(&self) -> usize;

    /// Fix the variable at `round_index` at a given `challenge` point. Mutates
    /// `self` to be the bookeeping table for the new MLE.
    ///
    /// If the new MLE becomes fully bound, returns the evaluation of the fully
    /// bound Mle.
    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>>;

    /// Fix the iterated variable at `indexed_bit_index` with a given challenge
    /// `point`. Mutates `self`` to be the bookeeping table for the new MLE.  If
    /// the new MLE becomes fully bound, returns the evaluation of the fully
    /// bound MLE in the form of a [Claim].
    ///
    /// # Panics
    /// If `indexed_bit_index` does not correspond to a
    /// `MleIndex::Iterated(indexed_bit_index)` in `mle_indices`.
    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<Claim<Self::F>>;

    /// Mutates the `MleIndices` that are `Iterated` and turns them into
    /// `IndexedBit` with the bit index being determined from `curr_index`.
    /// Returns the `curr_index + number of IndexedBits now in the
    /// MleIndices`.
    fn index_mle_indices(&mut self, curr_index: usize) -> usize;

    /// The layer_id of the layer that this MLE belongs to.
    fn get_layer_id(&self) -> LayerId;

    /// Whether the MLE has been indexed.
    fn indexed(&self) -> bool;

    /// Get the associated enum that this MLE is a part of ([MleEnum::Dense] or [MleEnum::Zero]).
    fn get_enum(self) -> MleEnum<Self::F>;
}

/// Trait that allows a type to be serialized into an [Mle], and yield [MleRef]s
/// TODO!(add a derive `MleAble`` macro that generates code for `FromIterator`,
/// `IntoIterator` and creates associated functions for yielding appropriate
/// `MleRefs`)
pub trait MleAble<F> {
    /// The particular representation that is convienent for an `MleAble`; most
    /// of the time it will be a `[Vec<F>; Size]` array
    type Repr: Send + Sync + Clone + Debug;

    /// The iterator that is used in order to iterate through the contents or data
    /// of the [MleAble].
    type IntoIter<'a>: Iterator<Item = Self>
    where
        Self: 'a;

    /// Get the evaluations of each of the points on the boolean hypercube.
    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F>;

    /// Convert into the [MleAble] repr from an iterator.
    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr;

    /// Iterate through the contents of the bookkeeping table.
    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_>;

    /// The number of variables associated with the MLE.
    fn num_vars(items: &Self::Repr) -> usize;
}

/// Represents all the possible types of indices for an [Mle].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum MleIndex<F> {
    /// A "selector" bit for fixed MLE access.
    Fixed(bool),

    /// An "unbound" bit that iterates over the contents of the MLE.
    Iterated,

    /// An "unbound" bit where the particular b_i in the larger expression has
    /// been set
    IndexedBit(usize),

    /// An index that has been bound to a random challenge by the sumcheck
    /// protocol.
    Bound(F, usize),
}

impl<F: FieldExt> MleIndex<F> {
    /// If `self` is an `Iterated` bit, turns it into an `IndexedBit(bit)`.
    /// Otherwise, `self` is not modified.
    ///
    /// TODO(Makis): We need a better name for this method!
    pub fn index_index(&mut self, bit: usize) {
        if let MleIndex::Iterated = self {
            *self = Self::IndexedBit(bit)
        }
    }

    /// If `self` is an `IndexedBit(idx)`, bind it to `chal`, i.e. turn it into
    /// a `Bound(chal, idx)` variant. Otherwise, `self` is not modified.
    pub fn bind_index(&mut self, chal: F) {
        if let MleIndex::IndexedBit(bit) = self {
            *self = Self::Bound(chal, *bit)
        }
    }

    /// Evaluate this `MleIndex`.
    /// * `Fixed(bit)` evaluates to `0` or `1` when `bit` is `false` or `true`
    ///   respectively.
    /// * `Iterated` bits are *not* evaluated, i.e. `None` is returned.
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

        // Iterated index.
        assert_eq!(MleIndex::<Fr>::Iterated.val(), None);

        // Indexed bit index.
        assert_eq!(MleIndex::<Fr>::IndexedBit(42).val(), None);
    }

    #[test]
    fn test_mle_index_bind() {
        // An `Fixed` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Fixed(true);
        mle_index.bind_index(Fr::from(17));
        assert_eq!(mle_index, MleIndex::<Fr>::Fixed(true));

        // An `Iterated` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Iterated;
        mle_index.bind_index(Fr::from(17));
        assert_eq!(mle_index, MleIndex::<Fr>::Iterated);

        // An `IndexedBit` index gets bound.
        let mut mle_index: MleIndex<Fr> = MleIndex::IndexedBit(42);
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
        mle_index.index_index(17);
        assert_eq!(mle_index, MleIndex::<Fr>::Fixed(true));

        // An `Iterated` index becomes `IndexedBit`.
        let mut mle_index: MleIndex<Fr> = MleIndex::Iterated;
        mle_index.index_index(17);
        assert_eq!(mle_index, MleIndex::<Fr>::IndexedBit(17));

        // An `IndexedBit` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::IndexedBit(42);
        mle_index.index_index(17);
        assert_eq!(mle_index, MleIndex::<Fr>::IndexedBit(42));

        // An `Bound` index remains unaffected.
        let mut mle_index: MleIndex<Fr> = MleIndex::Bound(Fr::from(17), 42);
        mle_index.index_index(37);
        assert_eq!(mle_index, MleIndex::<Fr>::Bound(Fr::from(17), 42));
    }
}

//! An MLE is a MultiLinearExtention that contains a more complex type (i.e. T, or (T, T) or ExampleStruct)

use core::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::claims::{Claim, wlx_eval::ClaimMle};
use crate::layer::LayerId;
use remainder_shared_types::FieldExt;

use self::mle_enum::MleEnum;
use dyn_clonable::*;
/// Contains default dense implementation of Mle
pub mod dense;

pub mod mle_enum;
pub mod zero;

pub mod betavalues;
pub mod evals;

//TODO!(Maybe this type needs PartialEq, could be easily implemented with a random id...)
///The trait that defines how a semantic Type (T) and a MultiLinearEvaluation containing field elements (F) interact.
/// T should always be a composite type containing Fs. For example (F, F) or a struct containing Fs.
///
/// If you want to construct an Mle, or use an Mle for some non-cryptographic computation (e.g. wit gen) then
/// you should always use the iterator adaptors IntoIterator and FromIterator, this is to ensure that the semantic ordering within T is always consistent.
#[clonable]
pub trait Mle<F>: Clone
where
    //+ CanonicalSerialize + CanonicalDeserialize,
    // + FromIterator<T>,
    F: FieldExt,
{
    ///Get the log_2 size of the WHOLE mle
    fn num_iterated_vars(&self) -> usize;
    ///Get the padded set of evaluations over the boolean hypercube; Useful for constructing the input layer
    fn get_padded_evaluations(&self) -> Vec<F>;

    fn set_prefix_bits(&mut self, new_bits: Option<Vec<MleIndex<F>>>);

    fn get_prefix_bits(&self) -> Option<Vec<MleIndex<F>>>;
}

///MleRef keeps track of an Mle and the fixed indices of the Mle to be used in an expression
pub trait MleRef: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de> {
    ///The Field Element this MleRef refers to
    type F: FieldExt;

    ///Gets reference to the current bookkeeping tables
    fn bookkeeping_table(&self) -> &[Self::F];

    ///Get claim that this MleRef Represents
    fn mle_indices(&self) -> &[MleIndex<Self::F>];

    fn original_mle_indices(&self) -> &Vec<MleIndex<Self::F>>;

    fn original_bookkeeping_table(&self) -> &[Self::F];

    // ///Moves the claim by adding the new_claims to the left of the originals
    // fn relabel_mle_indices(&mut self, new_claims: &[MleIndex<Self::F>]);

    fn push_mle_indices(&mut self, new_indices: &[MleIndex<Self::F>]);

    ///Number of variables the Mle this is a reference to is over
    fn num_vars(&self) -> usize;

    ///Number of original variables, not mutated
    fn original_num_vars(&self) -> usize;

    ///Fix the variable at round_index at a given challenge point, mutates self to be the bookeeping table for the new Mle.
    /// If the Mle is fully bound will return the evaluation of the fully bound Mle
    fn fix_variable(&mut self, round_index: usize, challenge: Self::F) -> Option<Claim<Self::F>>;

    /// Fix the iterated variable at `indexed_bit_index` with a given challenge
    /// `point`. Mutates self to be the bookeeping table for the new Mle.  If
    /// the Mle is fully bound will return the evaluation of the fully bound
    /// Mle.
    /// # Panics
    /// if `indexed_bit_index` does not correspond to a
    /// `MleIndex::Iterated(indexed_bit_index)` in `mle_indices`.
    fn fix_variable_at_index(
        &mut self,
        indexed_bit_index: usize,
        point: Self::F,
    ) -> Option<Claim<Self::F>>;

    ///Mutate the MleIndices that are Iterated and turn them into IndexedBit with the bit index being determined from curr_index.
    /// Returns the curr_index + the number of IndexedBits now in the MleIndices
    fn index_mle_indices(&mut self, curr_index: usize) -> usize;

    /// The layer_id of the layer that this MLE belongs to
    fn get_layer_id(&self) -> LayerId;

    /// get whether mle has been indexed
    fn indexed(&self) -> bool;

    fn get_enum(self) -> MleEnum<Self::F>;
}

///Trait that allows a type to be serialized into an Mle, and yield MleRefs
/// TODO!(add a derive MleAble macro that generates code for FromIterator, IntoIterator
/// and creates associated functions for yielding appropriate MleRefs)
pub trait MleAble<F> {
    ///The particular representation that is convienent for an MleAble, most of the time it will be a \[Vec<F>; Size\] array
    type Repr: Send + Sync + Clone + Debug;

    type IntoIter<'a>: Iterator<Item = Self>
    where
        Self: 'a;

    fn get_padded_evaluations(items: &Self::Repr) -> Vec<F>;

    fn from_iter(iter: impl IntoIterator<Item = Self>) -> Self::Repr;

    fn to_iter(items: &Self::Repr) -> Self::IntoIter<'_>;

    fn num_vars(items: &Self::Repr) -> usize;
}

///The Enum that represents the possible indices for an MLE
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub enum MleIndex<F> {
    ///A Selector bit for fixed MLE access
    Fixed(bool),
    ///An unbound bit that iterates over the contents of the MLE
    Iterated,
    ///An unbound bit where the particular b_i in the larger expression has been set
    IndexedBit(usize),
    ///an index that has been bound to a random challenge by the sumcheck protocol
    Bound(F, usize),
}

impl<F: FieldExt> MleIndex<F> {
    ///Turns this MleIndex into an IndexedBit variant if it's an Iterated variant
    pub fn index_index(&mut self, bit: usize) {
        if matches!(self, MleIndex::Iterated) {
            *self = Self::IndexedBit(bit)
        }
    }

    ///Bind an indexed bit to a challenge
    pub fn bind_index(&mut self, chal: F) {
        if let MleIndex::IndexedBit(bit) = self {
            *self = Self::Bound(chal, *bit)
        }
    }

    ///Evaluate this MleIndex
    pub fn val(&self) -> Option<F> {
        match self {
            MleIndex::Fixed(bit) => {
                if *bit {
                    Some(F::one())
                } else {
                    Some(F::zero())
                }
            }
            MleIndex::Bound(chal, _) => Some(*chal),
            _ => None,
        }
    }
}

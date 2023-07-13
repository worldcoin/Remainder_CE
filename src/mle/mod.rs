//! An MLE is a MultiLinearExtention that contains a more complex type (i.e. T, or (T, T) or ExampleStruct)


use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::fmt::Debug;

use std::ops::Index;

use crate::FieldExt;
use crate::layer::Claim;
use crate::mle::dense::BetaError;

///Contains default dense implementation of Mle
pub mod dense;

//TODO!(Maybe this type needs PartialEq, could be easily implemented with a random id...)
///The trait that defines how a semantic Type (T) and a MultiLinearEvaluation containing field elements (F) interact.
/// T should always be a composite type containing Fs. For example (F, F) or a struct containing Fs.
///
/// If you want to construct an Mle, or use an Mle for some non-cryptographic computation (e.g. wit gen) then
/// you should always use the iterator adaptors IntoIterator and FromIterator, this is to ensure that the semantic ordering within T is always consistent.
pub trait Mle<F, T>
where
    Self: Clone + Debug + CanonicalSerialize + CanonicalDeserialize,
    // + IntoIterator<Item = T>
    // + FromIterator<T>,
    F: FieldExt,
    //TODO!(Define MLEable trait + derive)
    T: Send + Sync + MleAble<F>,
{
    ///MleRef keeps track of an Mle and the fixed indices of the Mle to be used in an expression
    type MleRef: MleRef;

    ///Underlying MultiLinearExtention implementation
    type MultiLinearExtention: IntoIterator<Item = F>;

    ///Get number of variables of the Mle which is equivalent to the log_2 of the size of the MLE
    fn num_vars(&self) -> usize;
}

///MleRef keeps track of an Mle and the fixed indices of the Mle to be used in an expression
pub trait MleRef: Debug + Clone + Send + Sync {
    ///Type of Mle that this is a reference to
    type Mle: Index<usize, Output = Self::F>;

    ///The Field Element this MleRef refers to
    type F: FieldExt;

    ///Gets Mle that this is a reference to
    fn mle_owned(&self) -> Self::Mle;

    ///Gets reference to Mle
    fn mle(&self) -> &[Self::F];

    ///Get claim that this MleRef Represents
    fn mle_indices(&self) -> &[MleIndex<Self::F>];

    ///Moves the claim by adding the new_claims to the left of the originals
    fn relabel_mle_indices(&mut self, new_claims: &[MleIndex<Self::F>]);

    ///Number of variables the Mle this is a reference to is over
    fn num_vars(&self) -> usize;

    /// Initialize beta table 
    fn initialize_beta(
        &mut self,
        layer_claims: &Claim<Self::F>,
    ) -> Result<(), BetaError>;

    /// Update the beta table given random challenge at round j of sumcheck
    fn beta_update(
        &mut self,
        layer_claims: &Claim<Self::F>,
        round_index: usize,
        challenge: Self::F,
    ) -> Result<(), BetaError>;

    ///Fix the variable at round_index at a given challenge point, mutates self to be the bookeeping table for the new Mle.
    /// If the Mle is fully bound will return the evaluation of the fully bound Mle
    fn fix_variable(
        &mut self,
        round_index: usize,
        challenge: Self::F,
    ) -> Option<(Self::F, Vec<MleIndex<Self::F>>)>;

    ///Mutate the MleIndices that are Iterated and turn them into IndexedBit with the bit index being determined from curr_index.
    /// Returns the curr_index + the number of IndexedBits now in the MleIndices
    fn index_mle_indices(&mut self, curr_index: usize) -> usize;

    /// Get the current indices behind this MLE
    fn get_mle_indices(&self) -> &[MleIndex<Self::F>];

    /// The layer_id of the layer that this MLE belongs to
    fn get_layer_id(&self) -> Option<usize>;
}

///Trait that allows a type to be serialized into an Mle, and yield MleRefs
/// TODO!(add a derive MleAble macro that generates code for FromIterator, IntoIterator 
/// and creates associated functions for yielding appropriate MleRefs)
pub trait MleAble<F: FieldExt> {
    ///The particular representation that is convienent for an MleAble, most of the time it will be a \[Vec<F>; Size\] array
    type Repr: Send + Sync + Clone + Debug + CanonicalDeserialize + CanonicalSerialize;
}

///The Enum that represents the possible indices for an MLE
#[derive(Clone, Debug, PartialEq)]
pub enum MleIndex<F: FieldExt> {
    ///A Selector bit for fixed MLE access
    Fixed(bool),
    ///An unbound bit that iterates over the contents of the MLE
    Iterated,
    ///An unbound bit where the particular b_i in the larger expression has been set
    IndexedBit(usize),
    ///an index that has been bound to a random challenge by the sumcheck protocol
    Bound(F),
}

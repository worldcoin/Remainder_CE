//! An MLE is a MultiLinearExtention that contains a more complex type (i.e. T, or (T, T) or ExampleStruct)

use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::fmt::Debug;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator};
use std::ops::Index;

use crate::FieldExt;

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
    T: Send + Sync,
{
    ///MleRef keeps track of an Mle and the fixed indices of the Mle to be used in an expression
    type MleRef: MleRef;

    ///Underlying MultiLinearExtention implementation
    type MultiLinearExtention: IntoIterator<Item = F>;

    ///Gets underlying MultilinearExtention
    fn mle(&self) -> &Self::MultiLinearExtention;

    ///Gets default MleRef to be put into an expression
    fn mle_ref(&'_ self) -> Self::MleRef;

    ///Constructor that creates an Mle given a MultiLinearExtention
    fn new(mle: Self::MultiLinearExtention) -> Self;

    ///Get number of variables of the Mle which is equivalent to the log_2 of the size of the MLE
    fn num_vars(&self) -> usize;
}

///MleRef keeps track of an Mle and the fixed indices of the Mle to be used in an expression
pub trait MleRef: Clone + Send + Sync {
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

    fn fix_variable(&mut self, round_index: usize, challenge: Self::F);
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

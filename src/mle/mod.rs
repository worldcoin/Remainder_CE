//! An MLE is a MultiLinearExtention that contains a more complex type (i.e. T, or (T, T) or ExampleStruct)

use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::fmt::Debug;

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
    Self: Clone
        + Debug
        + CanonicalSerialize
        + CanonicalDeserialize
        + IntoIterator<Item = T>
        + FromIterator<T>,
    F: FieldExt,
    //TODO!(Define MLEable trait + derive)
    T: Send + Sync,
{
    ///MleRef keeps track of an Mle and the fixed indicies of the Mle to be used in an expression
    type MleRef<'a>: MleRef
    where
        Self: 'a;

    ///Underlying MultiLinearExtention implementation
    type MultiLinearExtention: MultilinearExtension<F>;

    ///Gets underlying MultilinearExtention
    fn mle(&self) -> &Self::MultiLinearExtention;

    ///Gets default MleRef to be put into an expression
    fn mle_ref<'a>(&'a self) -> Self::MleRef<'a>;

    ///Constructor that creates an Mle given a MultiLinearExtention
    fn new(mle: Self::MultiLinearExtention) -> Self;

    ///Get number of variables of the Mle which is equivalent to the log_2 of the size of the MLE
    fn num_vars(&self) -> usize;
}

///MleRef keeps track of an Mle and the fixed indicies of the Mle to be used in an expression
pub trait MleRef {
    ///Type of Mle that this is a reference to
    type Mle;

    ///Gets Mle that this is a reference to
    fn mle_owned(&self) -> Self::Mle;

    ///Gets reference to Mle
    fn mle(&self) -> &Self::Mle;

    ///Get claim that this MleRef Represents
    fn claim(&self) -> &[Option<bool>];

    ///Moves the claim by adding the new_claims to the left of the originals
    fn relabel_claim(&mut self, new_claims: &[Option<bool>]);
}

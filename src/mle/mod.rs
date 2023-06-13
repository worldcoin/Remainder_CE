//! An MLE is a MultiLinearExtention that contains a more complex type (i.e. T, or (T, T) or ExampleStruct)

use core::fmt::Debug;
use ark_poly::{MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::FieldExt;

///TODO!(Add detailed documentation to this type (theory + examples))
pub trait Mle<F, T>
where
    Self: Clone
        + Debug
        + CanonicalSerialize
        + CanonicalDeserialize
        + IntoIterator<Item = T>,
    F: FieldExt,
    //TODO!(Define MLEable trait + derive)
    T: Send + Sync,
{
    ///MleRef keeps track of an Mle and the fixed indicies of the Mle to be used in an expression
    type MleRef: MleRef;

    ///Underlying MultiLinearExtention implementation
    type MultiLinearExtention: MultilinearExtension<F>;

    ///Gets underlying MultilinearExtention
    fn mle(&self) -> Self::MultiLinearExtention;

    ///Gets default MleRef to be put into an expression
    fn mle_ref(&self) -> Self::MleRef;

    ///Constructor that creates an Mle given a MultiLinearExtention
    fn new(mle: Self::MultiLinearExtention) -> Self;

    ///Get number of variables of the Mle which is equivalent to the log_2 of the size of the MLE
    fn num_vars(&self) -> usize;
}

///MleRef keeps track of an Mle and the fixed indicies of the Mle to be used in an expression
pub trait MleRef {
    ///Parent Mle Type
    type Mle;

    ///Gets Mle that this is a reference to
    fn mle(&self) -> Self::Mle;

    ///Get claim that this MleRef Represents
    fn claim(&self) -> &[Option<bool>];

    ///Moves the claim by adding the new_claims to the left of the originals
    fn relabel_claim(&mut self, new_claims: &[Option<bool>]);
}

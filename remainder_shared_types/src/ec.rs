use crate::Field;

pub trait CurveExt
where
    Self: pasta_curves::arithmetic::CurveExt<Scalar: Field, Base: Field, Affine: CurveAffine>,
{
}

impl<C: pasta_curves::arithmetic::CurveExt<Scalar: Field, Base: Field, Affine: CurveAffine>>
    CurveExt for C
{
}

pub trait CurveAffine: pasta_curves::arithmetic::CurveAffine<Scalar: Field, Base: Field> {}

impl<C: pasta_curves::arithmetic::CurveAffine<Scalar: Field, Base: Field>> CurveAffine for C {}

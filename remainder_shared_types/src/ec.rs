use crate::FieldExt;

pub trait CurveExt
where
    Self: pasta_curves::arithmetic::CurveExt<Scalar: FieldExt, Base: FieldExt, Affine: CurveAffine>,
{
}

impl<
        C: pasta_curves::arithmetic::CurveExt<Scalar: FieldExt, Base: FieldExt, Affine: CurveAffine>,
    > CurveExt for C
{
}

pub trait CurveAffine:
    pasta_curves::arithmetic::CurveAffine<Scalar: FieldExt, Base: FieldExt>
{
}

impl<C: pasta_curves::arithmetic::CurveAffine<Scalar: FieldExt, Base: FieldExt>> CurveAffine for C {}

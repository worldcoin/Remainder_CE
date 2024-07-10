pub trait CurveExt: pasta_curves::arithmetic::CurveExt {}

impl<C: pasta_curves::arithmetic::CurveExt> CurveExt for C {}

pub trait CurveAffine: pasta_curves::arithmetic::CurveAffine {}

impl<C: pasta_curves::arithmetic::CurveAffine> CurveAffine for C {}
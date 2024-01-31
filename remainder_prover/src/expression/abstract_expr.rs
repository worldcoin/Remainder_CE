use std::{
    cmp::max,
    fmt::Debug,
    ops::{Add, Mul, Neg, Sub},
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    mle::{beta::*, dense::DenseMleRef, MleIndex, MleRef},
    sumcheck::MleError,
};

use remainder_shared_types::FieldExt;

use super::generic_expr::ExpressionType;


/// Abstract Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbstractExpression;
impl<F: FieldExt> ExpressionType<F> for AbstractExpression {
    type Container = Vec<F>;
}
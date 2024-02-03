use std::fmt::Debug;
use serde::{Deserialize, Serialize};

use remainder_shared_types::FieldExt;

use super::generic_expr::ExpressionType;


/// Abstract Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbstractExpression;
impl<F: FieldExt> ExpressionType<F> for AbstractExpression {
    type Container = Vec<F>;
    type MleRefs = ();
}
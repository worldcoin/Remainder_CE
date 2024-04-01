use std::fmt::Debug;
use serde::{Deserialize, Serialize};

use remainder_shared_types::FieldExt;

use super::generic_expr::ExpressionType;


/// Abstract Expression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AbstractExpr;
impl<F: FieldExt> ExpressionType<F> for AbstractExpr {
    type MLENodeRepr = Vec<F>;
    type MleVec = ();
}

//  comments for Phase II:
//  This will be the the circuit "pre-data" stage
//  will take care of building a prover expression
//  building the most memory efficient denseMleRefs dictionaries, etc.
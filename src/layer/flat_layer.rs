//!A layer following the basic GKR layer design
//! todo!(Add some math here with the template of a GKR layer)

use crate::{FieldExt, mle::dense::DenseMle, expression::ExpressionStandard};

use super::{Layer, Claim};

///A layer following the basic GKR layer design
pub struct FlatLayer<F: FieldExt> {
    id: usize,
    expression: Option<ExpressionStandard<F>>,
    round_index: usize,
}

impl<F: FieldExt> ConcreteLayer<F> for FlatLayer<F> {
    type Mle = DenseMle<F, F>;

    type Expression = ExpressionStandard<F>;

    fn new(id: usize) -> Self {
        Self {
            id,
            expression: None,
            round_index: 0,
        }
    }

    fn get_claim(&self) -> Option<Claim<F>> {
        todo!()
    }

    fn get_all_claims(&self) -> Option<Vec<(usize, Claim<F>)>> {
        todo!()
    }

    fn get_expression(&self) -> Self::Expression {
        todo!()
    }

    fn get_mles(&self) -> Vec<Self::Mle> {
        todo!()
    }

    fn get_id(&self) -> usize {
        self.id
    }
}
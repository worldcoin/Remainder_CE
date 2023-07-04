use crate::FieldExt;
use thiserror::Error;

use super::{Claim, Layer};

#[derive(Error, Debug, Clone)]
enum LayerError {
    #[error("The Layer has not finished the sumcheck protocol")]
    SumCheckNotComplete
}

///Take in a layer that has completed the sumcheck protocol and return a list of claims on the next layer
fn get_claims<'a, F: FieldExt>(layer: &'a impl Layer<F>) -> Result<Vec<Claim<'a, F>>, LayerError> {
    todo!()
}
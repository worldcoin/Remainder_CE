use serde::{Deserialize, Serialize};

use crate::FieldExt;

/// A claim contains a `point` \in F^n along with the `result` \in F that an
/// associated layer MLE is expected to evaluate to. In other words, if `W : F^n
/// -> F` is the MLE, then the claim asserts: `W(point) == result`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct Claim<F: FieldExt> {
    /// The point in F^n where the layer MLE is to be evaluated on.
    point: Vec<F>,
    /// The expected result of evaluating this layer's MLE on `point`.
    result: F,
}

impl<F: FieldExt> Claim<F> {
    /// Constructs a new `Claim`
    pub fn new(point: Vec<F>, result: F) -> Self {
        Self { point, result }
    }

    /// Returns the length of the `point` vector.
    pub fn get_num_vars(&self) -> usize {
        self.point.len()
    }

    /// Returns the point vector in F^n.
    pub fn get_point(&self) -> &Vec<F> {
        &self.point
    }

    /// Returns the expected result.
    pub fn get_result(&self) -> F {
        self.result
    }
}

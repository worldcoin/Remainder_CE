use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{layer::LayerId, mle::AbstractMle};

use super::MleIndex;

/// A version of [crate::mle::dense::DenseMle] used by the Verifier.
/// A [VerifierMle] stores a fully bound MLE along with its evaluation.
/// It is used to represent the leaves of an `Expression<F, VerifierExpr>`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Hash)]
#[serde(bound = "F: Field")]
pub struct VerifierMle<F: Field> {
    /// Layer whose data this MLE is a subset of.
    layer_id: LayerId,

    /// A list of bound indices.
    var_indices: Vec<MleIndex>,

    /// The evaluation of this MLE when variables are bound according
    /// `var_indices`.
    eval: F,
}

impl<F: Field> AbstractMle for VerifierMle<F> {
    /// Returns the MLE indices of this MLE.
    fn mle_indices(&self) -> &[MleIndex] {
        &self.var_indices
    }

    fn set_mle_indices(&mut self, new_indices: Vec<MleIndex>) {
        self.var_indices = new_indices
    }

    /// Returns the [LayerId] of this MLE.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: Field> VerifierMle<F> {
    /// Constructor for the [VerifierMle] using layer_id and the
    /// MLE indices that will go into this MLE. Additionally includes
    /// the eval, which is the evaluation of the fully bound MLE.
    pub fn new(layer_id: LayerId, var_indices: Vec<MleIndex>, eval: F) -> Self {
        Self {
            layer_id,
            var_indices,
            eval,
        }
    }

    /// Returns the num_vars of this MLE.
    pub fn num_vars(&self) -> usize {
        self.var_indices.len()
    }

    /// Returns the fully bound value of this MLE.
    pub fn value(&self) -> F {
        self.eval
    }
}

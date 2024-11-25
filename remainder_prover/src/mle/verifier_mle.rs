use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::layer::LayerId;

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
    var_indices: Vec<MleIndex<F>>,

    /// The evaluation of this MLE when variables are bound according
    /// `var_indices`.
    eval: F,
}

impl<F: Field> VerifierMle<F> {
    /// Constructor for the [VerifierMle] using layer_id and the
    /// MLE indices that will go into this MLE. Additionally includes
    /// the eval, which is the evaluation of the fully bound MLE.
    pub fn new(layer_id: LayerId, var_indices: Vec<MleIndex<F>>, eval: F) -> Self {
        Self {
            layer_id,
            var_indices,
            eval,
        }
    }

    /// Returns the layer_id of this MLE.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Returns the num_vars of this MLE.
    pub fn num_vars(&self) -> usize {
        self.var_indices.len()
    }

    /// Returns the MLE indices of this MLE.
    pub fn var_indices(&self) -> &[MleIndex<F>] {
        &self.var_indices
    }

    /// Returns the fully bound value of this MLE.
    pub fn value(&self) -> F {
        self.eval
    }
}

//!Utilities involving the claims a layer makes

///The claim aggregator that uses wlx evaluations
pub mod wlx_eval;

use remainder_shared_types::{
    input_layer::InputLayer,
    layer::{Layer, LayerId},
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    layer::{combine_mle_refs::CombineMleRefError, LayerError},
    prover::GKRError,
};

#[derive(Error, Debug, Clone)]
///Errors to do with aggregating and collecting claims
pub enum ClaimError {
    #[error("The Layer has not finished the sumcheck protocol")]
    ///The Layer has not finished the sumcheck protocol
    SumCheckNotComplete,
    #[error("MLE indices must all be fixed")]
    ///MLE indices must all be fixed
    ClaimMleIndexError,
    #[error("Layer ID not assigned")]
    ///Layer ID not assigned
    LayerMleError,
    #[error("MLE within MleRef has multiple values within it")]
    ///MLE within MleRef has multiple values within it
    MleRefMleError,
    #[error("Error aggregating claims")]
    ///Error aggregating claims
    ClaimAggroError,
    #[error("Should be evaluating to a sum")]
    ///Should be evaluating to a sum
    ExpressionEvalError,
    #[error("All claims in a group should agree on the number of variables")]
    ///All claims in a group should agree on the number of variables
    NumVarsMismatch,
    #[error("All claims in a group should agree the destination layer field")]
    ///All claims in a group should agree the destination layer field
    LayerIdMismatch,
    #[error("Error while combining mle refs in order to evaluate challenge point")]
    ///Error while combining mle refs in order to evaluate challenge point
    MleRefCombineError(CombineMleRefError),
}

pub use remainder_shared_types::claims::Claim;
pub use remainder_shared_types::claims::ClaimAndProof;

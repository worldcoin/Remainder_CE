//! A layer is a combination of multiple MLEs with an expression.

pub mod combine_mle_refs;
pub mod gate;
pub mod identity_gate;
pub mod layer_enum;
pub mod matmult;
pub mod product;
pub mod regular_layer;

use std::{collections::HashSet, fmt::Debug};

use layer_enum::{LayerEnum, VerifierLayerEnum};
use product::PostSumcheckLayer;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    claims::{Claim, ClaimError},
    expression::{circuit_expr::CircuitMle, expr_errors::ExpressionError},
    layouter::layouting::CircuitMap,
    sumcheck::InterpError,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    FieldExt,
};

/// Errors to do with working with a type implementing [Layer].
#[derive(Error, Debug, Clone)]
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    /// Layer isn't ready to prove
    LayerNotReady,
    #[error("Error with underlying expression: {0}")]
    /// Error with underlying expression: {0}
    ExpressionError(#[from] ExpressionError),
    #[error("Error with aggregating curr layer")]
    /// Error with aggregating curr layer
    AggregationError,
    #[error("Error with getting Claim: {0}")]
    /// Error with getting Claim
    ClaimError(#[from] ClaimError),
    #[error("Error with verifying layer: {0}")]
    /// Error with verifying layer
    VerificationError(#[from] VerificationError),
    #[error("InterpError: {0}")]
    /// InterpError
    InterpError(#[from] InterpError),
    #[error("Transcript Error: {0}")]
    /// Transcript Error
    TranscriptError(#[from] TranscriptReaderError),
}

/// Errors to do with verifying a layer while working with a type implementing
/// [VerifierLayer].
#[derive(Error, Debug, Clone)]
pub enum VerificationError {
    #[error("The sum of the first evaluations do not equal the claim")]
    /// The sum of the first evaluations do not equal the claim
    SumcheckStartFailed,

    #[error("The sum of the current rounds evaluations do not equal the previous round at a random point")]
    /// The sum of the current rounds evaluations do not equal the previous round at a random point
    SumcheckFailed,

    #[error("The final rounds evaluations at r do not equal the oracle query")]
    /// The final rounds evaluations at r do not equal the oracle query
    FinalSumcheckFailed,

    #[error("The Oracle query does not match the final claim")]
    /// The Oracle query does not match the final claim
    GKRClaimCheckFailed,

    #[error(
        "The Challenges generated during sumcheck don't match the claims in the given expression"
    )]
    ///The Challenges generated during sumcheck don't match the claims in the given expression
    ChallengeCheckFailed,

    /// Error with underlying expression: {0}
    #[error("Error with underlying expression: {0}")]
    ExpressionError(#[from] ExpressionError),

    /// Error while reading the transcript proof.
    #[error("Error while reading the transcript proof")]
    TranscriptError(#[from] TranscriptReaderError),

    /// Interpolation Error.
    #[error("Interpolation Error: {0}")]
    InterpError(#[from] InterpError),
}

/// A layer is the smallest component of the GKR protocol.
///
/// Each `Layer` is a sub-protocol that takes in some `Claim` and creates a proof
/// that the `Claim` is correct
pub trait Layer<F: FieldExt> {
    /// Gets this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Tries to prove `claim` for this layer.
    ///
    /// In the process of proving, it mutates itself binding the variables
    /// of the expression that define the layer.
    ///
    /// If successful, the proof is implicitly included in the modified
    /// transcript.
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError>;

    /// Initialize the sumcheck round by setting the beta table, computing the number of rounds, etc.
    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError>;

    /// Return the evaluations of the univariate polynomial generated during this round of sumcheck.
    ///
    /// This must be called with a steadily incrementing round_index & with a securely generated challenge.
    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError>;

    /// Mutate the underlying bookkeeping tables to "bind" the given `challenge` to the bit.
    /// labeled with that `round_index`.
    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError>;

    /// The list of sumcheck rounds this layer will prove, by index.
    fn sumcheck_round_indices(&self) -> Vec<usize>;

    /// The maximum degree for any univariate in the sumcheck protocol.
    fn max_degree(&self) -> usize;

    /// Get the [PostSumcheckLayer] for a certain layer, which is a struct which represents
    /// the fully bound layer.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F>;
}

/// A circuit-description counterpart of the GKR [Layer] trait.
pub trait CircuitLayer<F: FieldExt> {
    /// The associated type that the verifier uses to work with a layer of this
    /// kind.
    type VerifierLayer: VerifierLayer<F> + Debug + Serialize + for<'a> Deserialize<'a>;

    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Tries to verify `claim` for this layer and returns a [VerifierLayer]
    /// with a fully bound and evaluated expression.
    ///
    /// The proof is implicitly included in the `transcript`.
    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError>;

    /// The list of sumcheck rounds this layer will prove, by index.
    fn sumcheck_round_indices(&self) -> Vec<usize>;

    /// Turns this Circuit Layer into a Verifier Layer
    fn into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError>;

    /// Gets the [PostSumcheckLayer] for this layer.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>>;

    /// The maximum degree for any univariate in the sumcheck protocol.
    fn max_degree(&self) -> usize;

    /// Label the MLE indices, starting from the `start_index` by
    /// converting [MleIndex::Iterated] to [MleIndex::IndexedBit].
    fn index_mle_indices(&mut self, start_index: usize);

    /// Given the [CircuitMle]s of which outputs are expected of this layer, compute the data
    /// that populates these bookkeeping tables and mutate the circuit map to reflect this.
    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&CircuitMle<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) -> bool;

    /// The Circuit MLEs that make up the leaves of the expression in this layer.
    fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>>;

    /// Given a [CircuitMap], turn this [CircuitLayer] into a ProverLayer.
    fn into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F>;
}

/// A verifier counterpart of a GKR [Layer] trait.
pub trait VerifierLayer<F: FieldExt> {
    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
/// The location of a layer within the GKR circuit
pub enum LayerId {
    /// An Mle located in the input layer
    Input(usize),
    /// A layer between the output layer and input layers
    Layer(usize),
    /// A layer representing values sampled from the verifier via Fiat-Shamir
    VerifierChallengeLayer(usize),
}

impl LayerId {
    /// Gets a new LayerId which represents a layerid of the same type but with an incremented id number
    pub fn next(&self) -> LayerId {
        match self {
            LayerId::Input(id) => LayerId::Input(id + 1),
            LayerId::Layer(id) => LayerId::Layer(id + 1),
            LayerId::VerifierChallengeLayer(id) => LayerId::VerifierChallengeLayer(id + 1),
        }
    }

    /// Gets a new LayerId which represents a layerid of the same type but with an incremented id number.
    /// Mutates self to store the next layer id.
    pub fn get_and_inc(&mut self) -> LayerId {
        let ret = self.clone();
        self.increment_self();
        ret
    }

    /// Mutates self to store the next layer id.
    pub fn increment_self(&mut self) {
        match self {
            LayerId::Input(ref mut id) => {
                *id += 1;
            }
            LayerId::Layer(ref mut id) => {
                *id += 1;
            }
            LayerId::VerifierChallengeLayer(ref mut id) => {
                *id += 1;
            }
        };
    }
}

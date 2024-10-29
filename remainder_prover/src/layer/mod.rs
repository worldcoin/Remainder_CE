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
    claims::{Claim, ClaimError, RawClaim},
    expression::expr_errors::ExpressionError,
    layouter::{context::CircuitBuildingContext, layouting::CircuitMap},
    mle::mle_description::MleDescription,
    sumcheck::InterpError,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, TranscriptReaderError, VerifierTranscript},
    Field,
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
    /// Incorrect number of variable bindings
    #[error("Layer {0} requires {1} variable bindings, but {2} were provided")]
    NumVarsMismatch(LayerId, usize, usize),
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
pub trait Layer<F: Field> {
    /// Gets this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Initialize this layer and perform any necessary pre-computation: beta
    /// table, number of rounds, etc.
    fn initialize(&mut self, claim_point: &[F]) -> Result<(), LayerError>;

    /// Tries to prove `claim` for this layer.
    ///
    /// In the process of proving, it mutates itself binding the variables
    /// of the expression that define the layer.
    ///
    /// If successful, the proof is implicitly included in the modified
    /// transcript.
    fn prove(
        &mut self,
        claim: RawClaim<F>,
        transcript: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError>;

    /// Return the evaluations of the univariate polynomial generated during this round of sumcheck.
    ///
    /// This must be called with a steadily incrementing round_index & with a securely generated challenge.
    fn compute_round_sumcheck_message(&mut self, round_index: usize) -> Result<Vec<F>, LayerError>;

    /// Mutate the underlying bookkeeping tables to "bind" the given `challenge` to the bit.
    /// labeled with that `round_index`.
    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError>;

    /// The list of sumcheck rounds this layer will prove, by index.
    fn sumcheck_round_indices(&self) -> Vec<usize>;

    /// The maximum degree for any univariate in the sumcheck protocol.
    fn max_degree(&self) -> usize;

    /// Get the [PostSumcheckLayer] for a certain layer, which is a struct which represents
    /// the fully bound layer.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F>;

    /// Generates and returns all claims that this layer makes onto previous
    /// layers.
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError>;
}

/// A circuit-description counterpart of the GKR [Layer] trait.
pub trait LayerDescription<F: Field> {
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
        claim: RawClaim<F>,
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError>;

    /// The list of sumcheck rounds this layer will prove, by index.
    fn sumcheck_round_indices(&self) -> Vec<usize>;

    /// Turns this [LayerDescription] into a [VerifierLayer] by taking the
    /// `sumcheck_bindings` and `claim_point` and inserting them into the
    /// expression to become a verifier expression.
    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError>;

    /// Gets the [PostSumcheckLayer] for this layer.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>>;

    /// The maximum degree for any univariate in the sumcheck protocol.
    fn max_degree(&self) -> usize;

    /// Label the MLE indices, starting from the `start_index` by
    /// converting [MleIndex::Free] to [MleIndex::IndexedBit].
    fn index_mle_indices(&mut self, start_index: usize);

    /// Given the [MleDescription]s of which outputs are expected of this layer, compute the data
    /// that populates these bookkeeping tables and mutate the circuit map to reflect this.
    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&MleDescription<F>>,
        circuit_map: &mut CircuitMap<F>,
    );

    /// The [MleDescription]s that make up the leaves of the expression in this layer.
    fn get_circuit_mles(&self) -> Vec<&MleDescription<F>>;

    /// Given a [CircuitMap], turn this [LayerDescription] into a ProverLayer.
    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F>;
}

/// A verifier counterpart of a GKR [Layer] trait.
pub trait VerifierLayer<F: Field> {
    /// Returns this layer's ID.
    fn layer_id(&self) -> LayerId;

    /// Get the claims that this layer makes on other layers.
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
/// The location of a layer within the GKR circuit
pub enum LayerId {
    /// An Mle located in the input layer
    Input(usize),
    /// A layer representing values sampled from the verifier via Fiat-Shamir
    FiatShamirChallengeLayer(usize),
    /// A layer between the output layer and input layers
    Layer(usize),
}

/// Implement Display for LayerId, so that we can use it in error messages
impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerId::Input(id) => write!(f, "Input Layer {}", id),
            LayerId::Layer(id) => write!(f, "Layer {}", id),
            LayerId::FiatShamirChallengeLayer(id) => {
                write!(f, "Fiat-Shamir Challenge Layer {}", id)
            }
        }
    }
}

impl LayerId {
    /// Creates a new LayerId representing an input layer.
    pub fn next_input_layer_id() -> Self {
        LayerId::Input(CircuitBuildingContext::next_input_layer_id())
    }

    /// Creates a new LayerId representing a layer.
    pub fn next_layer_id() -> Self {
        LayerId::Layer(CircuitBuildingContext::next_layer_id())
    }

    /// Creates a new LayerId representing a Fiat-Shamir challenge layer.
    pub fn next_fiat_shamir_challenge_layer_id() -> Self {
        LayerId::FiatShamirChallengeLayer(CircuitBuildingContext::next_fiat_shamir_challenge_layer_id())
    }

    /// Returns the underlying usize if self is a variant of type Input, otherwise panics.
    pub fn get_raw_input_layer_id(&self) -> usize {
        match self {
            LayerId::Input(id) => *id,
            _ => panic!("Expected LayerId::Input, found {:?}", self),
        }
    }

    /// Returns the underlying usize if self is a variant of type Input, otherwise panics.
    pub fn get_raw_layer_id(&self) -> usize {
        match self {
            LayerId::Layer(id) => *id,
            _ => panic!("Expected LayerId::Layer, found {:?}", self),
        }
    }
}

use remainder_shared_types::{
    halo2curves::CurveExt,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    input_layer::ec_input::ECInputLayer,
    layer::{ec_layer::ECLayer, LayerError, LayerId},
    prover::GKRError,
};

use super::YieldClaim;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: CurveExt")]
pub struct HyraxClaim<C: CurveExt> {
    _marker: std::marker::PhantomData<C>,
}

/// A wrapper struct that represents the `Claim` a `ClaimAggregator`
/// generates and the proof that this claim is correct
#[derive(Clone, Debug)]
pub struct ECClaimAndProof<C: CurveExt, P> {
    /// The `Claim` that was proven
    pub claim: HyraxClaim<C>,
    /// The proof that the `Claim` is correct
    pub proof: P,
}

pub trait ECClaimAggregator<C: CurveExt>
where
    C::ScalarExt: FieldExt,
{
    ///The struct the claim aggregator takes in.
    ///
    /// Is typically composed of the `Claim` struct and additional information
    type Claim: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>;

    ///The proof that a verifier can use to check that the aggregation was done correctly
    type AggregationProof: std::fmt::Debug + Clone + Serialize + for<'a> Deserialize<'a>;

    ///The Layer this claim aggregator takes claims from, and aggregates claims for
    type Layer: ECLayer<C>;

    ///The InputLayer this claim aggregator aggregates claims for
    type InputLayer: ECInputLayer<C>;

    ///Creates an empty ClaimAggregator, ready to track claims
    fn new() -> Self;

    ///Takes in claims to track and later aggregate
    fn add_claims(&mut self, layer: &impl YieldClaim<Self::Claim>) -> Result<(), LayerError>;

    ///Retrieves claims for aggregation
    fn get_claims(&self, layer_id: LayerId) -> Option<&[Self::Claim]>;

    ///Takes in some claims from the prover and aggregates them into one claim
    ///
    /// Extracts additional information from the `Layer` the claims are made on if neccessary.
    ///
    /// Adds any communication to the F-S Transcript
    fn prover_aggregate_claims(
        &self,
        layer: &Self::Layer,
        transcript_writer: &mut impl ECProverTranscript<C::Affine>,
    ) -> Result<ECClaimAndProof<C, Self::AggregationProof>, GKRError>;

    ///Takes in some claims from the prover and aggregates them into one claim
    ///
    /// Extracts additional information from the `InputLayer` the claims are made on if neccessary.
    ///
    /// Adds any communication to the F-S Transcript
    fn prover_aggregate_claims_input(
        &self,
        layer: &Self::InputLayer,
        transcript_writer: &mut impl ECProverTranscript<C::Affine>,
    ) -> Result<ECClaimAndProof<C, Self::AggregationProof>, GKRError>;

    ///Reads an AggregationProof from the Transcript and uses it to verify the aggregation of some claims.
    fn verifier_aggregate_claims(
        &self,
        layer_id: LayerId,
        transcript_reader: &mut impl ECVerifierTranscript<C::Affine>,
    ) -> Result<HyraxClaim<C>, GKRError>;
}

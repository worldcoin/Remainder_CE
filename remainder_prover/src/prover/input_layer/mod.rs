//! Trait for dealing with InputLayer

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::*;
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;
pub mod combine_input_layers;
pub mod enum_input_layer;
pub mod ligero_input_layer;
pub mod public_input_layer;
pub mod random_input_layer;

use crate::{
    claims::{wlx_eval::{get_num_wlx_evaluations, ClaimGroup, YieldWLXEvals, ENABLE_PRE_FIX}, Claim, ClaimError}, layer::{
        combine_mle_refs::pre_fix_mle_refs, LayerError, LayerId
    }, mle::{dense::DenseMle, mle_enum::MleEnum, MleIndex, MleRef}, prover::ENABLE_OPTIMIZATION, sumcheck::evaluate_at_a_point
};

use self::enum_input_layer::InputLayerEnum;

use ark_std::{end_timer, start_timer};

#[derive(Error, Clone, Debug)]
pub enum InputLayerError {
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
    #[error("failed to verify random input layer")]
    RandomInputVerificationFailed,
    #[error("Error during interaction with the transcript.")]
    TranscriptError(TranscriptReaderError),
    #[error("Challenge or consumed element did not match the expected value.")]
    TranscriptMatchError,
}

use log::{debug, info, trace, warn};
///Trait for dealing with the InputLayer
pub trait InputLayer<F: FieldExt> {
    /// The struct that contains the commitment to the contents of the input_layer
    type Commitment: Serialize + for<'a> Deserialize<'a>;

    /// The struct that contains the opening proof
    type OpeningProof: Serialize + for<'a> Deserialize<'a>;

    /// Generates a commitment
    /// 
    /// Can mutate self to cache useful information
    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    ///Appends the commitment to the F-S Transcript
    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    );

    ///Appends the commitment to the F-S Transcript
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError>;

    /// Generates a proof of polynomial evaluation at the point
    /// in the `Claim`
    /// 
    /// Appends any communication to the transcript
    fn open(
        &self,
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, InputLayerError>;

    /// Verifies the evaluation at the point in the `Claim` relative to the
    /// polynomial commitment using the opening proof
    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError>;

    ///Returns the `LayerId` of this layer
    fn layer_id(&self) -> &LayerId;

    ///Returns the contents of this `InputLayer` as an
    /// owned `DenseMle`
    fn get_padded_mle(&self) -> DenseMle<F, F>;
}

///Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    ///Creates a new InputLayer from a flat mle
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self;
}

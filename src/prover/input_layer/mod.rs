//! Trait for dealing with InputLayer

use ark_std::log2;
use itertools::Itertools;
use lcpc_2d::{FieldExt, fs_transcript::halo2_remainder_transcript::{Transcript, TranscriptError}};
use thiserror::Error;
pub mod combine_input_layers;
pub mod ligero_input_layer;
pub mod public_input_layer;
pub mod random_input_layer;
pub mod enum_input_layer;

use crate::{layer::{LayerId, Claim}, mle::{Mle, dense::DenseMle, MleIndex}, utils::argsort, prover::input_layer_faje::invert_mle_bookkeeping_table};

#[derive(Error, Clone, Debug)]
pub enum InputLayerError {
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
}


///Trait for dealing with the InputLayer
pub trait InputLayer<F: FieldExt> {
    type Transcript: Transcript<F>;
    type Commitment;
    type OpeningProof;

    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    fn append_commitment_to_transcript(commitment: &Self::Commitment, transcript: &mut Self::Transcript) -> Result<(), TranscriptError>;

    fn open(&self, transcript: &mut Self::Transcript, claim: Claim<F>) -> Result<Self::OpeningProof, InputLayerError>;

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), InputLayerError>;

    fn layer_id(&self) -> &LayerId;
}

///Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    ///Creates a new InputLayer from a flat mle
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self;
}
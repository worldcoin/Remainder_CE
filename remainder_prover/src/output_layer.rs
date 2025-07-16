//! An GKR Output Layer is a "virtual layer". This means it is not assigned a
//! fresh [LayerId]. Instead, it is associated with some other
//! intermediate/input layer whose [LayerId] it inherits. The MLE it stores is a
//! restriction of an MLE defining its associated layer.

use itertools::Itertools;
use remainder_shared_types::{
    transcript::{TranscriptReaderError, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    claims::Claim,
    layer::{LayerError, LayerId},
};

use crate::{
    claims::ClaimError,
    layouter::layouting::CircuitMap,
    mle::{
        dense::DenseMle, mle_description::MleDescription, mle_enum::MleEnum,
        verifier_mle::VerifierMle, zero::ZeroMle, Mle, MleIndex,
    },
};

use anyhow::{anyhow, Result};

// Unit tests for Output Layers.
#[cfg(test)]
pub mod tests;

/// Output layers are "virtual layers" in the sense that they are not assigned a
/// separate [LayerId]. Instead they are associated with the ID of an existing
/// intermediate/input layer on which they generate claims for.
/// Contains an [MleEnum] which can be either a [DenseMle] or a [ZeroMle].
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "F: Field")]
pub struct OutputLayer<F: Field> {
    mle: MleEnum<F>,
}

/// Required for output layer shenanigans within `layout`
impl<F: Field> From<DenseMle<F>> for OutputLayer<F> {
    fn from(value: DenseMle<F>) -> Self {
        Self {
            mle: MleEnum::Dense(value),
        }
    }
}

impl<F: Field> From<ZeroMle<F>> for OutputLayer<F> {
    fn from(value: ZeroMle<F>) -> Self {
        Self {
            mle: MleEnum::Zero(value),
        }
    }
}

impl<F: Field> OutputLayer<F> {
    /// Returns the MLE contained within.
    pub fn get_mle(&self) -> &MleEnum<F> {
        &self.mle
    }

    /// Generate a new [OutputLayer] from a [ZeroMle].
    pub fn new_zero(zero_mle: ZeroMle<F>) -> Self {
        Self {
            mle: MleEnum::Zero(zero_mle),
        }
    }

    /// If the MLE is fully-bound, returns its evaluation.
    /// Otherwise, it returns an [super::OutputLayerError].
    pub fn value(&self) -> Result<F> {
        match &self.mle {
            MleEnum::Dense(_) => unimplemented!(),
            MleEnum::Zero(zero_mle) => {
                if !zero_mle.is_fully_bounded() {
                    return Err(anyhow!(OutputLayerError::MleNotFullyBound));
                }

                Ok(F::ZERO)
            }
        }
    }

    /// Returns the [LayerId] of the intermediate/input layer that this output
    /// layer is associated with.
    pub fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    /// Number of free variables.
    pub fn num_free_vars(&self) -> usize {
        self.mle.num_free_vars()
    }

    /// Whether the output layer is fully bounded
    pub fn is_fully_bounded(&self) -> bool {
        self.mle.is_fully_bounded()
    }

    /// Fix the variables of this output layer to random challenges sampled
    /// from the transcript.
    /// Expects `self.num_free_vars()` challenges.
    pub fn fix_layer(&mut self, challenges: &[F]) -> Result<()> {
        let bits = self.mle.index_mle_indices(0);
        if bits != challenges.len() {
            return Err(anyhow!(LayerError::NumVarsMismatch(
                self.mle.layer_id(),
                bits,
                challenges.len(),
            )));
        }
        (0..bits)
            .zip(challenges.iter())
            .for_each(|(bit, challenge)| {
                self.mle.fix_variable(bit, *challenge);
            });
        debug_assert!(self.is_fully_bounded());
        Ok(())
    }

    /// Extract a claim on this output layer by extracting the bindings from the fixed variables.
    pub fn get_claim(&mut self) -> Result<Claim<F>> {
        if !self.mle.is_fully_bounded() {
            return Err(anyhow!(LayerError::ClaimError(ClaimError::MleRefMleError)));
        }

        let mle_indices: Result<Vec<F>> = self
            .mle
            .mle_indices()
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(anyhow!(LayerError::ClaimError(ClaimError::MleRefMleError)))
            })
            .collect();

        let claim_value = self.mle.first();

        Ok(Claim::new(
            mle_indices?,
            claim_value,
            self.mle.layer_id(),
            self.mle.layer_id(),
        ))
    }
}

/// The circuit description type for the defaul Output Layer consisting of an
/// MLE.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash)]
#[serde(bound = "F: Field")]
pub struct OutputLayerDescription<F: Field> {
    /// The metadata of this MLE: indices and associated layer.
    pub mle: MleDescription<F>,

    /// Whether this is an MLE that is supposed to evaluate to zero.
    is_zero: bool,
}

impl<F: Field> OutputLayerDescription<F> {
    /// Generate an output layer containing a verifier equivalent of a
    /// [DenseMle], with a given `layer_id` and `mle_indices`.
    pub fn new_dense(_layer_id: LayerId, _mle_indices: &[MleIndex<F>]) -> Self {
        // We do not allow `DenseMle`s at this point.
        unimplemented!()
    }

    /// Generate an output layer containing a verifier equivalent of a
    /// [ZeroMle], with a given `layer_id` and `mle_indices`.
    pub fn new_zero(layer_id: LayerId, mle_indices: &[MleIndex<F>]) -> Self {
        Self {
            mle: MleDescription::new(layer_id, mle_indices),
            is_zero: true,
        }
    }

    /// Determine whether the MLE Output layer contains an MLE whose
    /// coefficients are all 0.
    pub fn is_zero(&self) -> bool {
        self.is_zero
    }

    /// Label the MLE indices in this layer, starting from the start_index.
    pub fn index_mle_indices(&mut self, start_index: usize) {
        self.mle.index_mle_indices(start_index);
    }

    /// Convert this into the prover view of an output layer, using the [CircuitMap].
    pub fn into_prover_output_layer(&self, circuit_map: &CircuitMap<F>) -> OutputLayer<F> {
        let output_mle = circuit_map.get_data_from_circuit_mle(&self.mle).unwrap();
        let prefix_bits = self.mle.prefix_bits();
        let prefix_bits_mle_index = prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .collect();

        if self.is_zero {
            // Ensure that the calculated output MLE is all zeroes.
            if !(output_mle.iter().all(|val| val == F::ZERO)) {
                println!(
                    "WARNING: MLE for output layer {} is not zero",
                    self.mle.layer_id()
                );
                dbg!(output_mle.iter().take(10).collect_vec());
            }
            ZeroMle::new(
                output_mle.num_vars(),
                Some(prefix_bits_mle_index),
                self.layer_id(),
            )
            .into()
        } else {
            DenseMle::new_with_prefix_bits(output_mle.clone(), self.layer_id(), prefix_bits).into()
        }
    }
}

impl<F: Field> OutputLayerDescription<F> {
    /// Returns the [LayerId] of the intermediate/input layer that his output
    /// layer is associated with.
    pub fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    /// Retrieve the MLE evaluations from the transcript and fix the variables
    /// of this output layer to random challenges sampled from the transcript.
    /// Returns a description of the layer ready to be used by the verifier.
    pub fn retrieve_mle_from_transcript_and_fix_layer(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierOutputLayer<F>> {
        // We do not yet handle DenseMle.
        assert!(self.is_zero());

        let num_evals = 1;

        let evals = transcript_reader.consume_elements("Output layer MLE evals", num_evals)?;

        if evals != vec![F::ZERO] {
            return Err(anyhow!(VerifierOutputLayerError::NonZeroEvalForZeroMle));
        }

        let bits = self.mle.num_free_vars();

        let mut mle = self.mle.clone();

        // Evaluate each output MLE at a random challenge point.
        for bit in 0..bits {
            let challenge = transcript_reader.get_challenge("Challenge on the output layer")?;
            mle.fix_variable(bit, challenge);
        }

        debug_assert_eq!(mle.num_free_vars(), 0);

        let verifier_output_layer =
            VerifierOutputLayer::new_zero(self.mle.layer_id(), mle.var_indices(), F::ZERO);

        Ok(verifier_output_layer)
    }
}

/// The verifier counterpart type for the defaul Output Layer consisting of an
/// MLE.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: Field")]
pub struct VerifierOutputLayer<F: Field> {
    /// A description of this layer's fully-bound MLE.
    mle: VerifierMle<F>,

    /// Whether this layer's MLE is supposed to evaluate to zero.
    is_zero: bool,
}

impl<F: Field> VerifierOutputLayer<F> {
    /// Generate an output layer containing a verifier equivalent of a
    /// [DenseMle], with a given `layer_id` and `mle_indices`.
    pub fn new_dense(_layer_id: LayerId, _mle_indices: &[MleIndex<F>]) -> Self {
        // We do not allow `DenseMle`s at this point.
        unimplemented!()
    }

    /// Generate an output layer containing a verifier equivalent of a
    /// [ZeroMle], with a given `layer_id`, `mle_indices` and `value`.
    pub fn new_zero(layer_id: LayerId, mle_indices: &[MleIndex<F>], value: F) -> Self {
        Self {
            mle: VerifierMle::new(layer_id, mle_indices.to_vec(), value),
            is_zero: true,
        }
    }

    /// Determine whether this output layer represents an MLE
    /// whose coefficients are all 0.
    pub fn is_zero(&self) -> bool {
        self.is_zero
    }

    /// The number of variables used to represent the underlying MLE.
    pub fn num_vars(&self) -> usize {
        self.mle.num_vars()
    }
}

impl<F: Field> VerifierOutputLayer<F> {
    /// Returns the [LayerId] of the intermediate/input layer that this output
    /// layer is associated with.
    pub fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    /// Extract a claim on this output layer by extracting the bindings from the fixed variables.
    pub fn get_claim(&self) -> Result<Claim<F>> {
        // We do not support non-zero MLEs on Output Layers at this point!
        assert!(self.is_zero());

        let layer_id = self.layer_id();

        let prefix_bits: Vec<MleIndex<F>> = self
            .mle
            .var_indices()
            .iter()
            .filter(|index| matches!(index, MleIndex::Fixed(_bit)))
            .cloned()
            .collect();

        let claim_point: Vec<F> = self
            .mle
            .var_indices()
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(anyhow!(LayerError::ClaimError(ClaimError::MleRefMleError)))
            })
            .collect::<Result<Vec<_>>>()?;

        let num_vars = self.num_vars();
        let num_prefix_bits = prefix_bits.len();
        let num_free_vars = num_vars - num_prefix_bits;

        let claim_value = self.mle.value();

        // The verifier is expecting to receive a fully-bound [MleRef]. Start
        // with an unindexed MLE, index it, and then bound its variables.
        let mut claim_mle = MleEnum::Zero(ZeroMle::new(num_free_vars, Some(prefix_bits), layer_id));
        claim_mle.index_mle_indices(0);

        for mle_index in self.mle.var_indices().iter() {
            if let MleIndex::Bound(val, idx) = mle_index {
                claim_mle.fix_variable(*idx, *val);
            }
        }

        Ok(Claim::new(
            claim_point,
            claim_value,
            self.mle.layer_id(),
            self.mle.layer_id(),
        ))
    }
}

/// Errors to do with working with a type implementing [OutputLayer].
#[derive(Error, Debug, Clone)]
pub enum OutputLayerError {
    /// Expected fully-bound MLE.
    #[error("Expected fully-bound MLE")]
    MleNotFullyBound,
}

/// Errors to do with working with a type implementing [VerifierOutputLayer].
#[derive(Error, Debug, Clone)]
pub enum VerifierOutputLayerError {
    /// Prover sent a non-zero value for a ZeroMle.
    #[error("Prover sent a non-zero value for a ZeroMle")]
    NonZeroEvalForZeroMle,

    /// Transcript Reader Error during verification.
    #[error("Transcript Reader Error: {:0}", _0)]
    TranscriptError(#[from] TranscriptReaderError),
}

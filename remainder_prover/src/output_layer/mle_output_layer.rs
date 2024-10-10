//! The default implementation of a GKR Output Layer which uses an MLE type
//! to encode its data and meta-data.
//!
//! An GKR Output Layer is a "virtual layer". This means it is not assigned a
//! fresh [LayerId]. Instead, it is associated with some other
//! intermediate/input layer whose [LayerId] it inherits. The MLE it stores is a
//! restriction of an MLE defining its associated layer.

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::ClaimMle, ClaimError, YieldClaim},
    expression::{circuit_expr::MleDescription, verifier_expr::VerifierMle},
    layer::{LayerError, LayerId},
    layouter::layouting::CircuitMap,
    mle::{dense::DenseMle, mle_enum::MleEnum, zero::ZeroMle, Mle, MleIndex},
};

use super::{
    OutputLayer, OutputLayerDescription, OutputLayerError, VerifierOutputLayer,
    VerifierOutputLayerError,
};

/// The Prover's default type of an [crate::output_layer::OutputLayer].
/// Contains an [MleEnum] which can be either a [DenseMle] or a [ZeroMle].
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "F: Field")]
pub struct MleOutputLayer<F: Field> {
    mle: MleEnum<F>,
}

/// Required for output layer shenanigans within `layout`
impl<F: Field> From<DenseMle<F>> for MleOutputLayer<F> {
    fn from(value: DenseMle<F>) -> Self {
        Self {
            mle: MleEnum::Dense(value),
        }
    }
}

impl<F: Field> From<ZeroMle<F>> for MleOutputLayer<F> {
    fn from(value: ZeroMle<F>) -> Self {
        Self {
            mle: MleEnum::Zero(value),
        }
    }
}

impl<F: Field> MleOutputLayer<F> {
    /// Returns the MLE contained within. For PROVER use only!
    pub fn get_mle(&self) -> &MleEnum<F> {
        &self.mle
    }

    /// Generate a new [MleOutputLayer] from a [DenseMle].
    pub fn new_dense(_dense_mle: DenseMle<F>) -> Self {
        // We do not currently allow `DenseMle`s in the output.
        unimplemented!();
    }

    /// Generate a new [MleOutputLayer] from a [ZeroMle].
    pub fn new_zero(zero_mle: ZeroMle<F>) -> Self {
        Self {
            mle: MleEnum::Zero(zero_mle),
        }
    }

    /// If the MLE is fully-bound, returns its evaluation.
    /// Otherwise, it returns an [super::OutputLayerError].
    pub fn value(&self) -> Result<F, OutputLayerError> {
        match &self.mle {
            MleEnum::Dense(_) => unimplemented!(),
            MleEnum::Zero(zero_mle) => {
                if zero_mle.num_free_vars() != 0 {
                    return Err(OutputLayerError::MleNotFullyBound);
                }

                Ok(F::ZERO)
            }
        }
    }
}

impl<F: Field> OutputLayer<F> for MleOutputLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    fn fix_layer(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), crate::layer::LayerError> {
        let bits = self.mle.index_mle_indices(0);

        // Evaluate each output MLE at a random challenge point.
        for bit in 0..bits {
            let challenge = transcript_writer.get_challenge("Setting Output Layer Claim");
            self.mle.fix_variable(bit, challenge);
        }

        debug_assert_eq!(self.mle.num_free_vars(), 0);

        Ok(())
    }

    fn append_mle_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<F>) {
        transcript_writer.append_elements(
            "Output Layer MLE evals",
            &self.mle.iter().collect::<Vec<_>>(),
        );
    }
}

/// The circuit description type for the defaul Output Layer consisting of an
/// MLE.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash)]
#[serde(bound = "F: Field")]
pub struct MleOutputLayerDescription<F: Field> {
    /// The metadata of this MLE: indices and associated layer.
    pub mle: MleDescription<F>,

    /// Whether this is an MLE that is supposed to evaluate to zero.
    is_zero: bool,
}

impl<F: Field> MleOutputLayerDescription<F> {
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
    pub fn into_prover_output_layer(&self, circuit_map: &CircuitMap<F>) -> MleOutputLayer<F> {
        let output_mle = circuit_map.get_data_from_circuit_mle(&self.mle).unwrap();
        let prefix_bits = self.mle.prefix_bits();
        let prefix_bits_mle_index = prefix_bits
            .iter()
            .map(|bit| MleIndex::Fixed(*bit))
            .collect();

        if self.is_zero {
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

impl<F: Field> OutputLayerDescription<F> for MleOutputLayerDescription<F> {
    type VerifierOutputLayer = VerifierMleOutputLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    fn retrieve_mle_from_transcript_and_fix_layer(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierOutputLayer, VerifierOutputLayerError> {
        // We do not yet handle DenseMle.
        assert!(self.is_zero());

        let num_evals = 1;

        let evals = transcript_reader.consume_elements("Output Layer MLE evals", num_evals)?;

        if evals != vec![F::ZERO] {
            return Err(VerifierOutputLayerError::NonZeroEvalForZeroMle);
        }

        let bits = self.mle.num_free_vars();

        let mut mle = self.mle.clone();

        // Evaluate each output MLE at a random challenge point.
        for bit in 0..bits {
            let challenge = transcript_reader.get_challenge("Setting Output Layer Claim")?;
            mle.fix_variable(bit, challenge);
        }

        debug_assert_eq!(mle.num_free_vars(), 0);

        let verifier_output_layer =
            VerifierMleOutputLayer::new_zero(self.mle.layer_id(), mle.mle_indices(), F::ZERO);

        Ok(verifier_output_layer)
    }
}

/// The verifier counterpart type for the defaul Output Layer consisting of an
/// MLE.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: Field")]
pub struct VerifierMleOutputLayer<F: Field> {
    /// A description of this layer's fully-bound MLE.
    mle: VerifierMle<F>,

    /// Whether this layer's MLE is supposed to evaluate to zero.
    is_zero: bool,
}

impl<F: Field> VerifierMleOutputLayer<F> {
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

impl<F: Field> VerifierOutputLayer<F> for VerifierMleOutputLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for MleOutputLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        if self.mle.len() != 1 {
            return Err(LayerError::ClaimError(ClaimError::MleRefMleError));
        }

        let mle_indices: Result<Vec<F>, _> = self
            .mle
            .mle_indices()
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::MleRefMleError))
            })
            .collect();

        let claim_value = self.mle.first();

        Ok(vec![ClaimMle::new(
            mle_indices?,
            claim_value,
            None,
            Some(self.mle.layer_id()),
        )])
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for VerifierMleOutputLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        // We do not support non-zero MLEs on Output Layers at this point!
        assert!(self.is_zero());

        let layer_id = self.layer_id();

        let prefix_bits: Vec<MleIndex<F>> = self
            .mle
            .mle_indices()
            .iter()
            .filter(|index| matches!(index, MleIndex::Fixed(_bit)))
            .cloned()
            .collect();

        let claim_point: Vec<F> = self
            .mle
            .mle_indices()
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::MleRefMleError))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let num_vars = self.num_vars();
        let num_prefix_bits = prefix_bits.len();
        let num_free_vars = num_vars - num_prefix_bits;

        let claim_value = self.mle.value();

        // The verifier is expecting to receive a fully-bound [MleRef]. Start
        // with an unindexed MLE, index it, and then bound its variables.
        let mut claim_mle = MleEnum::Zero(ZeroMle::new(num_free_vars, Some(prefix_bits), layer_id));
        claim_mle.index_mle_indices(0);

        for mle_index in self.mle.mle_indices().iter() {
            if let MleIndex::Bound(val, idx) = mle_index {
                claim_mle.fix_variable(*idx, *val);
            }
        }

        Ok(vec![ClaimMle::new(
            claim_point,
            claim_value,
            None,
            Some(self.mle.layer_id()),
        )])
    }
}

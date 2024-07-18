//! The default implementation of a GKR Output Layer which uses an MLE type
//! to encode its data and meta-data.
//!
//! An GKR Output Layer is a "virtual layer". This means it is not assigned a
//! fresh [LayerId]. Instead, it is associated with some other
//! intermediate/input layer whose [LayerId] it inherits. The MLE it stores is a
//! restriction of an MLE defining its associated layer.

use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::ClaimMle, ClaimError, ProverYieldClaim, VerifierYieldClaim},
    expression::circuit_expr::CircuitMle,
    layer::{LayerError, LayerId},
    mle::{dense::DenseMle, mle_enum::MleEnum, zero::ZeroMle, Mle, MleIndex},
};

use super::{OutputLayer, OutputLayerError, VerifierOutputLayer, VerifierOutputLayerError};

/// The Prover's default type of an [crate::output_layer::OutputLayer].
/// Contains an [MleEnum] which can be either a [DenseMle] or a [ZeroMle].
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MleOutputLayer<F: FieldExt> {
    mle: MleEnum<F>,
}

impl<F: FieldExt> MleOutputLayer<F> {
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
                if zero_mle.num_iterated_vars() != 0 {
                    return Err(OutputLayerError::MleNotFullyBound);
                }

                Ok(F::ZERO)
            }
        }
    }
}

impl<F: FieldExt> OutputLayer<F> for MleOutputLayer<F> {
    type VerifierOutputLayer = VerifierMleOutputLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.mle.get_layer_id()
    }

    // TODO(Makis): Add method: If Dense(dense_mle), append all outputs to
    // transcript.

    fn fix_layer(
        &mut self,
        transcript_writer: &mut remainder_shared_types::transcript::TranscriptWriter<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) -> Result<(), crate::layer::LayerError> {
        let bits = self.mle.index_mle_indices(0);

        // Evaluate each output MLE at a random challenge point.
        for bit in 0..bits {
            let challenge = transcript_writer.get_challenge("Setting Output Layer Claim");
            self.mle.fix_variable(bit, challenge);
        }

        debug_assert_eq!(self.mle.num_iterated_vars(), 0);

        Ok(())
    }

    fn into_verifier_output_layer(&self) -> Self::VerifierOutputLayer {
        let mut mle = self.mle.clone();
        mle.index_mle_indices(0);

        let layer_id = mle.get_layer_id();
        let indices = mle.mle_indices();

        match self.mle {
            MleEnum::Dense(_) => Self::VerifierOutputLayer::new_dense(layer_id, indices),
            MleEnum::Zero(_) => Self::VerifierOutputLayer::new_zero(layer_id, indices),
        }
    }

    fn append_mle_to_transcript(
        &self,
        transcript_writer: &mut remainder_shared_types::transcript::TranscriptWriter<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) {
        transcript_writer.append_elements("Output Layer MLE evals", self.mle.bookkeeping_table());
    }
}

/// The default type of a verifier's Output Layer consisting of an MLE.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMleOutputLayer<F: FieldExt> {
    mle: CircuitMle<F>,
}

impl<F: FieldExt> VerifierMleOutputLayer<F> {
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
            mle: CircuitMle::new_zero(layer_id, mle_indices),
        }
    }
}

impl<F: FieldExt> VerifierOutputLayer<F> for VerifierMleOutputLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
    }

    fn fix_layer(
        &mut self,
        transcript_reader: &mut remainder_shared_types::transcript::TranscriptReader<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) -> Result<(), VerifierOutputLayerError> {
        let bits = self.mle.num_iterated_vars();
        dbg!(bits);

        // Evaluate each output MLE at a random challenge point.
        for bit in 0..bits {
            let challenge = transcript_reader.get_challenge("Setting Output Layer Claim")?;
            self.mle.fix_variable(bit, challenge);
        }

        debug_assert_eq!(self.mle.num_iterated_vars(), 0);

        Ok(())
    }

    fn retrieve_mle_from_transcript(
        &mut self,
        transcript_reader: &mut remainder_shared_types::transcript::TranscriptReader<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) -> Result<(), VerifierOutputLayerError> {
        let num_evals = if self.mle.is_zero() {
            1
        } else {
            self.mle.num_iterated_vars()
        };

        let evals = transcript_reader.consume_elements("Output Layer MLE evals", num_evals)?;

        if self.mle.is_zero() {
            if evals != vec![F::ZERO] {
                return Err(VerifierOutputLayerError::NonZeroEvalForZeroMle);
            }
        } else {
            unimplemented!();
        }

        Ok(())
    }
}

impl<F: FieldExt> ProverYieldClaim<F, ClaimMle<F>> for MleOutputLayer<F> {
    fn get_claims(
        &self,
        transcript_writer: &mut remainder_shared_types::transcript::TranscriptWriter<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        if self.mle.bookkeeping_table().len() != 1 {
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

        let claim_value = self.mle.bookkeeping_table()[0];
        transcript_writer.append("MleOutputLayer claim result", claim_value);

        Ok(vec![ClaimMle::new(
            mle_indices?,
            claim_value,
            None,
            Some(self.mle.layer_id()),
            Some(self.mle.clone()),
        )])
    }
}

impl<F: FieldExt> VerifierYieldClaim<F, ClaimMle<F>> for VerifierMleOutputLayer<F> {
    fn get_claims(
        &self,
        transcript_reader: &mut remainder_shared_types::transcript::TranscriptReader<
            F,
            impl remainder_shared_types::transcript::TranscriptSponge<F>,
        >,
    ) -> Result<Vec<ClaimMle<F>>, crate::layer::LayerError> {
        let layer_id = self.layer_id();

        let mle_indices: Vec<F> = self
            .mle
            .mle_indices()
            .iter()
            .map(|index| {
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::MleRefMleError))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let num_vars = mle_indices.len();

        let claim_value = transcript_reader.consume_element("MleOutputLayer claim result")?;

        // We do not yet handle the [DenseMle] case.
        if !self.mle.is_zero() {
            unimplemented!()
        }

        // The verifier is expecting to receive a fully-bound [MleRef].
        // Start with an iterated MLE, index it, and then bound its variables.
        let mut claim_mle = MleEnum::Zero(ZeroMle::new(num_vars, None, layer_id));
        claim_mle.index_mle_indices(0);
        for (idx, val) in mle_indices.iter().enumerate() {
            claim_mle.fix_variable(idx, *val);
        }

        Ok(vec![ClaimMle::new(
            mle_indices,
            claim_value,
            None,
            Some(self.mle.layer_id()),
            Some(claim_mle),
        )])
    }
}

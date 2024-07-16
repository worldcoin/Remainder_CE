//! The default implementation of a GKR Output Layer.

use remainder_shared_types::FieldExt;
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::ClaimMle, ProverYieldClaim, VerifierYieldClaim},
    expression::{circuit_expr::CircuitMle, verifier_expr::VerifierMle},
    layer::LayerId,
    mle::{dense::DenseMle, mle_enum::MleEnum, zero::ZeroMle, Mle},
};

use super::{OutputLayer, VerifierOutputLayer};

/// The default type of a prover's Output Layer consisting of an MLE.
/// Can be either a [DenseMle] or a [ZeroMle].
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MleOutputLayer<F: FieldExt> {
    mle: MleEnum<F>,
}

impl<F: FieldExt> MleOutputLayer<F> {
    /// Return a new [MleOutputLayer] containing a [DenseMle].
    pub fn new_dense(dense_mle: DenseMle<F>) -> Self {
        Self {
            mle: MleEnum::Dense(dense_mle),
        }
    }

    /// Return a new [MleOutputLayer] containing a [ZeroMle].
    pub fn new_zero(zero_mle: ZeroMle<F>) -> Self {
        Self {
            mle: MleEnum::Zero(zero_mle),
        }
    }
}

impl<F: FieldExt> OutputLayer<F> for MleOutputLayer<F> {
    type VerifierOutputLayer = VerifierMleOutputLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.mle.get_layer_id()
    }

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
        match self.mle {
            MleEnum::Dense(dense_mle) => Self::VerifierOutputLayer {
                mle: CircuitMle::new(
                    dense_mle.get_layer_id(),
                    dense_mle.mle_indices()),
            },
            MleEnum::Zero(zero_mle) => ,
        }
    }
}

/// The default type of a verifier's Output Layer consisting of an MLE.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMleOutputLayer<F: FieldExt> {
    mle: CircuitMle<F>,
}

impl<F: FieldExt> VerifierOutputLayer<F> for VerifierMleOutputLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.mle.layer_id()
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
        todo!()
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
        todo!()
    }
}

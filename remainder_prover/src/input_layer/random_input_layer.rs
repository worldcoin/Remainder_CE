//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    layer::LayerId,
    mle::{dense::DenseMle, mle_enum::MleEnum},
};

use super::{get_wlx_evaluations_helper, InputLayer, InputLayerError, VerifierInputLayer};
use crate::mle::Mle;

/// Represents a random input layer, where we generate random constants in the
/// form of coefficients of an MLE that we can use for packing constants.

pub struct RandomInputLayer<F: FieldExt> {
    mle: Vec<F>,
    pub(crate) layer_id: LayerId,
}

/// Verifier's description of a Random Input Layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierRandomInputLayer<F: FieldExt> {
    /// The ID of this Random Input Layer.
    layer_id: LayerId,

    /// The number of variables this Random Input Layer is on.
    num_bits: usize,

    _marker: PhantomData<F>,
}

impl<F: FieldExt> InputLayer<F> for RandomInputLayer<F> {
    type Commitment = Vec<F>;

    type VerifierInputLayer = VerifierRandomInputLayer<F>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        // We do not need to commit to the randomness, so we simply send it in
        // the clear.
        Ok(self.mle.clone())
    }

    fn append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        unimplemented!()
    }

    fn open(
        &self,
        _transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        _claim: Claim<F>,
    ) -> Result<(), super::InputLayerError> {
        // We do not have an opening proof because we did not commit to
        // anything. The MLE exists in the clear.
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id)
    }
}

impl<F: FieldExt> VerifierInputLayer<F> for VerifierRandomInputLayer<F> {
    type Commitment = Vec<F>;

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::Commitment, InputLayerError> {
        let num_evals = 1 << self.num_bits;
        Ok(transcript_reader.consume_elements("Random Layer Commitment", num_evals)?)
    }

    fn verify(
        &self,
        _: &Self::Commitment,
        claim: Claim<F>,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError> {
        // In order to verify, simply fix variable on each of the variables for
        // the point in `claim`. Check whether the single element left in the
        // bookkeeping table is equal to the claimed value in `claim`.
        let num_evals = 1 << self.num_bits;
        let mle_evals = transcript.get_challenges("Random Input Layer Challenges", num_evals)?;
        let mut mle_ref = DenseMle::<F>::new_from_raw(mle_evals, self.layer_id);
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_iterated_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            eval.ok_or(InputLayerError::RandomInputVerificationFailed)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };

        if eval.get_point() == claim.get_point() && eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            Err(InputLayerError::RandomInputVerificationFailed)
        }
    }
}

impl<F: FieldExt> RandomInputLayer<F> {
    /// Generates a random MLE of size `size` that is generated from the FS Transcript
    pub fn new(
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        size: usize,
        layer_id: LayerId,
    ) -> Self {
        let mle = transcript.get_challenges("Random Input Layer Challenges", size);
        Self { mle, layer_id }
    }

    /// Return the MLE stored in self as a DenseMle with the correct layer ID.
    pub fn get_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id)
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for RandomInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        get_wlx_evaluations_helper(
            self,
            claim_vecs,
            claimed_vals,
            claimed_mles,
            num_claims,
            num_idx,
        )
    }
}

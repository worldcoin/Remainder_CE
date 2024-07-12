//! A part of the input layer that is random and secured through F-S

use ark_std::log2;
use remainder_shared_types::{
    layer::LayerId,
    transcript::{ProverTranscript, TranscriptSponge, TranscriptWriter, VerifierTranscript},
    FieldExt,
};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    mle::{
        dense::DenseMle,
        evals::{Evaluations, MultilinearExtension},
        mle_enum::MleEnum,
    },
};

use super::{get_wlx_evaluations_helper, InputLayer, InputLayerError, MleInputLayer};
use crate::mle::Mle;

/// Represents a random input layer, where we generate random constants in the
/// form of coefficients of an MLE that we can use for packing constants.

pub struct RandomInputLayer<F: FieldExt> {
    mle: Vec<F>,
    pub(crate) layer_id: LayerId,
}

impl<F: FieldExt> InputLayer<F> for RandomInputLayer<F> {
    type Commitment = Vec<F>;

    type OpeningProof = ();

    type Error = InputLayerError;

    /// We do not need to commit to the randomness, so we simply send it in the clear.
    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.clone())
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError> {
        for challenge in commitment {
            let real_chal = transcript_reader
                .get_challenge("Getting RandomInput")
                .map_err(InputLayerError::TranscriptError)?;
            if *challenge != real_chal {
                return Err(InputLayerError::TranscriptMatchError);
            }
        }
        Ok(())
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn prover_append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        unimplemented!()
    }

    /// We do not have an opening proof because we did not commit to anything. The MLE
    /// exists in the clear.
    fn open(
        &self,
        _transcript: &mut impl ProverTranscript<F>,
        _claim: Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    /// In order to verify, simply fix variable on each of the variables for the point
    /// in `claim`. Check whether the single element left in the bookkeeping table is
    /// equal to the claimed value in `claim`.
    fn verify(
        commitment: &Self::Commitment,
        _opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        _transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F>::new_from_raw(commitment.to_vec(), LayerId::Input(0));
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

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }
}

impl<F: FieldExt> RandomInputLayer<F> {
    /// Generates a random MLE of size `size` that is generated from the FS Transcript
    pub fn new(
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        size: usize,
        layer_id: LayerId,
    ) -> Self {
        let mle = transcript.get_challenges("Getting Random Challenges", size);
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
            MultilinearExtension::new(self.mle.clone()),
            claim_vecs,
            claimed_vals,
            claimed_mles,
            num_claims,
            num_idx,
        )
    }
}

//! An input layer that is sent to the verifier in the clear

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    layer::LayerId,
    mle::{dense::DenseMle, mle_enum::MleEnum, MleRef},
};

use super::{get_wlx_evaluations_helper, InputLayer, InputLayerError, MleInputLayer};
use crate::mle::Mle;

/// An Input Layer in which the data is sent to the verifier
/// "in the clear" (i.e. without a commitment).
pub struct PublicInputLayer<F: FieldExt> {
    mle: DenseMle<F>,
    pub(crate) layer_id: LayerId,
}

impl<F: FieldExt> InputLayer<F> for PublicInputLayer<F> {
    type Commitment = Vec<F>;

    type OpeningProof = ();

    /// Because this is a public input layer, we do not need to commit to the MLE and the
    /// "commitment" is just the MLE itself.
    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.mle.clone())
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        transcript_writer.append_elements("Public Input Commitment", commitment);
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError> {
        let num_elements = commitment.len();
        let transcript_commitment = transcript_reader
            .consume_elements("Public Input Commitment", num_elements)
            .map_err(InputLayerError::TranscriptError)?;
        debug_assert_eq!(transcript_commitment, *commitment);
        Ok(())
    }

    /// We do not have an opening proof because we did not commit to anything. The MLE
    /// exists in the clear.
    fn open(
        &self,
        _: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        _: crate::claims::Claim<F>,
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
        _transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F>::new_from_raw(commitment.clone(), LayerId::Input(0), None);
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            debug_assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };

        if eval.get_point() == claim.get_point() && eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            Err(InputLayerError::PublicInputVerificationFailed)
        }
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        self.mle.clone()
    }
}

impl<F: FieldExt> MleInputLayer<F> for PublicInputLayer<F> {
    fn new(mle: DenseMle<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for PublicInputLayer<F> {
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

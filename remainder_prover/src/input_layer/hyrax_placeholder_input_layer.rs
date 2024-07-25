//! An input layer that is sent to the verifier in the clear

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    layer::LayerId,
    mle::{evals::MultilinearExtension, mle_enum::MleEnum},
};

use super::{InputLayer, InputLayerError, MleInputLayer};

/// An Input Layer in which the data is sent to the verifier
/// "in the clear" (i.e. without a commitment).
pub struct HyraxPlaceholderInputLayer<F: FieldExt> {
    pub mle: MultilinearExtension<F>,
    pub(crate) layer_id: LayerId,
}

impl<F: FieldExt> InputLayer<F> for HyraxPlaceholderInputLayer<F> {
    type Commitment = Vec<F>;

    type OpeningProof = ();

    type Error = InputLayerError;

    /// Because this is a public input layer, we do not need to commit to the MLE and the
    /// "commitment" is just the MLE itself.
    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.get_evals_vector().clone())
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn prover_append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        unimplemented!()
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn verifier_append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError> {
        unimplemented!()
    }

    /// We do not have an opening proof because we did not commit to anything. The MLE
    /// exists in the clear.
    fn open(
        &self,
        _: &mut impl ProverTranscript<F>,
        _: crate::claims::Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    /// In order to verify, simply fix variable on each of the variables for the point
    /// in `claim`. Check whether the single element left in the bookkeeping table is
    /// equal to the claimed value in `claim`.
    fn verify(
        _commitment: &Self::Commitment,
        _opening_proof: &Self::OpeningProof,
        _claim: Claim<F>,
        _transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<(), super::InputLayerError> {
        unimplemented!()
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }
}

impl<F: FieldExt> MleInputLayer<F> for HyraxPlaceholderInputLayer<F> {
    fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for HyraxPlaceholderInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        _claim_vecs: &[Vec<F>],
        _claimed_vals: &[F],
        _claimed_mles: Vec<MleEnum<F>>,
        _num_claims: usize,
        _num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        unimplemented!()
    }
}

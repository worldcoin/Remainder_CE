//! A part of the input layer that is random and secured through F-S




use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};


use crate::{
    claims::{
        wlx_eval::{YieldWLXEvals},
        Claim,
    },
    layer::{LayerId},
    mle::{dense::DenseMle, mle_enum::MleEnum, MleRef},
};



use super::{
    get_wlx_evaluations_helper, InputLayer, InputLayerError,
};

pub struct RandomInputLayer<F: FieldExt> {
    mle: Vec<F>,
    pub(crate) layer_id: LayerId,
}

impl<F: FieldExt> InputLayer<F> for RandomInputLayer<F> {
    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.clone())
    }

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
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

    fn prover_append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        unimplemented!()
    }

    fn open(
        &self,
        _transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        _claim: Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(
        commitment: &Self::Commitment,
        _opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        _transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), super::InputLayerError> {
        // println!("3, calling verify");
        let mut mle_ref =
            DenseMle::<F, F>::new_from_raw(commitment.to_vec(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            // println!("1, eval = {:#?}, claim = {:#?}", eval, claim);

            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };

        if eval.get_point() == claim.get_point() && eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            // println!("2, eval = {:#?}, claim = {:#?}", eval, claim);
            Err(InputLayerError::PublicInputVerificationFailed)
        }
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id, None)
    }
}

impl<F: FieldExt> RandomInputLayer<F> {
    ///Generates a random MLE of size `size` that is generated from the FS Transcript
    pub fn new(
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        size: usize,
        layer_id: LayerId,
    ) -> Self {
        let mle = transcript.get_challenges("Getting Random Challenges", size);
        Self { mle, layer_id }
    }

    pub fn get_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id, None)
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for RandomInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
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

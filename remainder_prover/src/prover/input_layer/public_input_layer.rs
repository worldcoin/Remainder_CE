//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use ark_std::{cfg_into_iter, end_timer, start_timer};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use tracing::{debug, info};

use crate::{
    claims::{wlx_eval::{get_num_wlx_evaluations, YieldWLXEvals, ENABLE_PRE_FIX}, Claim}, layer::{LayerError, LayerId}, mle::{dense::DenseMle, mle_enum::MleEnum, MleIndex, MleRef}, sumcheck::evaluate_at_a_point
};

use super::{enum_input_layer::InputLayerEnum, get_wlx_evaluations_helper, InputLayer, InputLayerError, MleInputLayer};

use rayon::prelude::{ParallelIterator, IntoParallelIterator};

///An Input Layer that is send to the verifier in the clear
pub struct PublicInputLayer<F: FieldExt> {
    mle: DenseMle<F, F>,
    pub(crate) layer_id: LayerId,
}

impl<F: FieldExt> InputLayer<F> for PublicInputLayer<F> {
    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.mle.clone())
    }

    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        transcript_writer.append_elements("Public Input Commitment", commitment);
    }

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError> {
        let num_elements = commitment.len();
        let transcript_commitment = transcript_reader
            .consume_elements("Public Input Commitment", num_elements)
            .map_err(|e| InputLayerError::TranscriptError(e))?;
        debug_assert_eq!(transcript_commitment, *commitment);
        Ok(())
    }

    fn open(
        &self,
        _: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        _: crate::claims::Claim<F>,
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
            DenseMle::<F, F>::new_from_raw(commitment.clone(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            debug_assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            // println!("1, eval = {:#?}, claim = {:#?}", eval, claim);
            // dbg!(&eval);
            // dbg!(&claim);
            (eval.ok_or(InputLayerError::PublicInputVerificationFailed)?).clone()
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
        self.mle.clone()
    }
}

impl<F: FieldExt> MleInputLayer<F> for PublicInputLayer<F> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
        }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for PublicInputLayer<F> {
        
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        get_wlx_evaluations_helper(self, claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx)
    }
}

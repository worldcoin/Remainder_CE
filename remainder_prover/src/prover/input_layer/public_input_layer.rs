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

use super::{enum_input_layer::InputLayerEnum, InputLayer, InputLayerError, MleInputLayer};

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
        let prep_timer = start_timer!(|| "Claim wlx prep");
        let mut mle_ref = self.get_padded_mle().clone().mle_ref();
        end_timer!(prep_timer);
        info!(
            "Wlx MLE len: {}",
            mle_ref.current_mle.get_evals_vector().len()
        );

        //fix variable hella times
        //evaluate expr on the mutated expr

        // get the number of evaluations
        mle_ref.index_mle_indices(0);
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);
        let chal_point = &claim_vecs[0];

        if ENABLE_PRE_FIX {
            if common_idx.is_some() {
                let common_idx = common_idx.unwrap();
                common_idx.iter().for_each(|chal_idx| {
                    if let MleIndex::IndexedBit(idx_bit_num) = mle_ref.mle_indices()[*chal_idx] {
                        mle_ref.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
                    }
                });
            }
        }

        debug!("Evaluating {num_evals} times.");

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            // let next_evals: Vec<F> = (num_claims..num_evals).into_iter()
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    // let new_chal: Vec<F> = (0..num_idx).into_iter()
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            // let evals: Vec<F> = (&claim_vecs).into_iter()
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                let mut fix_mle = mle_ref.clone();
                {
                    new_chal.into_iter().enumerate().for_each(|(idx, chal)| {
                        if let MleIndex::IndexedBit(idx_num) = fix_mle.mle_indices()[idx] {
                            fix_mle.fix_variable(idx_num, chal);
                        }
                    });
                    fix_mle.current_mle[0]
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        debug!("Returning evals:\n{:#?} ", wlx_evals);
        Ok(wlx_evals)

    }
}

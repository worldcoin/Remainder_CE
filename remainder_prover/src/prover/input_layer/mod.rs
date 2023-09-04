//! Trait for dealing with InputLayer

use ark_std::{log2, cfg_into_iter, cfg_iter};
use rayon::prelude::*;
use itertools::Itertools;
use remainder_shared_types::{FieldExt, transcript::{Transcript, TranscriptError}};
use thiserror::Error;
pub mod combine_input_layers;
pub mod ligero_input_layer;
pub mod public_input_layer;
pub mod random_input_layer;
pub mod enum_input_layer;

use crate::{layer::{LayerId, Claim, claims::ClaimError}, mle::{Mle, dense::DenseMle, MleIndex, MleRef}, utils::argsort, sumcheck::evaluate_at_a_point};

use self::enum_input_layer::InputLayerEnum;


#[derive(Error, Clone, Debug)]
pub enum InputLayerError {
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
}

///Trait for dealing with the InputLayer
pub trait InputLayer<F: FieldExt> {
    type Transcript: Transcript<F>;
    type Commitment;
    type OpeningProof;

    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    fn append_commitment_to_transcript(commitment: &Self::Commitment, transcript: &mut Self::Transcript) -> Result<(), TranscriptError>;

    fn open(&self, transcript: &mut Self::Transcript, claim: Claim<F>) -> Result<Self::OpeningProof, InputLayerError>;

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), InputLayerError>;

    fn layer_id(&self) -> &LayerId;

    fn get_padded_mle(&self) -> DenseMle<F, F>;

    fn compute_claim_wlx(
        &self,
        claims: &[Claim<F>],
    ) -> Result<Vec<F>, ClaimError> {
            let mut mle = self.get_padded_mle().clone().mle_ref();
            let num_claims = claims.len();
            let (claim_vecs, mut claimed_vals): (Vec<Vec<F>>, Vec<F>) = cfg_iter!(claims).cloned().unzip();
            let num_idx = claim_vecs[0].len();

            //fix variable hella times
            //evaluate expr on the mutated expr

            // get the number of evaluations
            let num_vars = mle.index_mle_indices(0);
            let num_evals = (num_vars) * (num_claims); 

            // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
            let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
                .map(|idx| {
                    // get the challenge l(idx)
                    let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                        .map(|claim_idx| {
                            let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                                .map(|claim| claim[claim_idx])
                                .collect();
                            let res = evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap();
                            res
                        })
                        .collect();

                    let mut fix_mle = mle.clone();
                    let eval = {
                        new_chal.into_iter().enumerate().for_each(
                            |(idx, chal)| {
                                fix_mle.fix_variable(idx, chal);
                            }
                        ); 
                        fix_mle.bookkeeping_table[0]
                    }; 
                    eval

                })
                .collect();

            // concat this with the first k evaluations from the claims to get num_evals evaluations
            claimed_vals.extend(&next_evals);
            let wlx_evals = claimed_vals.clone();
            Ok(wlx_evals)
    }

    /// Computes `l(r^{\star})`
    fn compute_aggregated_challenges(
        &self,
        claims: &[Claim<F>],
        rstar: F,
    ) -> Result<Vec<F>, ClaimError> {
        let (claim_vecs, _): (Vec<Vec<F>>, Vec<F>) = cfg_iter!(claims).cloned().unzip();
    
        if claims.is_empty() {
            return Err(ClaimError::ClaimAggroError);
        }
        let num_idx = claim_vecs[0].len();
    
        // get the claim (r = l(r*))
        let r: Vec<F> = cfg_into_iter!(0..num_idx)
            .map(|idx| {
                let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                    .map(|claim| claim[idx])
                    .collect();
                evaluate_at_a_point(&evals, rstar).unwrap()
            })
            .collect();
    
        Ok(r)
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript>;
}

///Adapter for InputLayerBuilder, implement for InputLayers that can be built out of flat MLEs
pub trait MleInputLayer<F: FieldExt>: InputLayer<F> {
    ///Creates a new InputLayer from a flat mle
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self;
}
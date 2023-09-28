//! Trait for dealing with InputLayer

use ark_std::{cfg_into_iter, cfg_iter};

use rayon::prelude::*;
use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};
use thiserror::Error;
pub mod combine_input_layers;
pub mod enum_input_layer;
pub mod ligero_input_layer;
pub mod public_input_layer;
pub mod random_input_layer;

use crate::{
    layer::{
        claims::{Claim, ClaimError, ClaimGroup},
        LayerId,
    },
    mle::{dense::DenseMle, MleRef},
    sumcheck::evaluate_at_a_point,
};

use self::enum_input_layer::InputLayerEnum;

#[derive(Error, Clone, Debug)]
pub enum InputLayerError {
    #[error("You are opening an input layer before generating a commitment!")]
    OpeningBeforeCommitment,
    #[error("failed to verify public input layer")]
    PublicInputVerificationFailed,
    #[error("failed to verify random input layer")]
    RandomInputVerificationFailed,
}

use log::{debug, info, trace, warn};
///Trait for dealing with the InputLayer
pub trait InputLayer<F: FieldExt> {
    type Transcript: Transcript<F>;
    type Commitment;
    type OpeningProof;

    fn commit(&mut self) -> Result<Self::Commitment, InputLayerError>;

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut Self::Transcript,
    ) -> Result<(), TranscriptError>;

    fn open(
        &self,
        transcript: &mut Self::Transcript,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, InputLayerError>;

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), InputLayerError>;

    fn layer_id(&self) -> &LayerId;

    fn get_padded_mle(&self) -> DenseMle<F, F>;

    fn compute_claim_wlx(&self, claims: &ClaimGroup<F>) -> Result<Vec<F>, ClaimError> {
        let mut mle = self.get_padded_mle().clone().mle_ref();
        let num_claims = claims.get_num_claims();
        let claim_vecs = claims.get_claim_points_matrix();
        let claimed_vals = claims.get_results();
        let num_idx = claims.get_num_vars();

        //fix variable hella times
        //evaluate expr on the mutated expr

        // get the number of evaluations
        let num_vars = mle.index_mle_indices(0);
        let num_evals = num_claims * num_vars;

        let mut degree_reduction = num_vars as i64;
        for j in 0..num_vars {
            for i in 1..num_claims {
                if claim_vecs[i][j] != claim_vecs[i - 1][j] {
                    degree_reduction -= 1;
                    break;
                }
            }
        }
        assert!(degree_reduction >= 0);

        // Evaluate the P(x) := W(l(x)) polynomial at deg(P) + 1
        // points. W : F^n -> F is a multi-linear polynomial on
        // `num_vars` variables and l : F -> F^n is a canonical
        // polynomial passing through `num_claims` points so its degree is
        // at most `num_claims - 1`. This imposes an upper
        // bound of `num_vars * (num_claims - 1)` to the degree of P.
        // However, the actual degree of P might be lower.
        // For any coordinate `i` such that all claims agree
        // on that coordinate, we can quickly deduce that `l_i(x)` is a
        // constant polynomial of degree zero instead of `num_claims -
        // 1` which brings down the total degree by the same amount.
        let num_evals =
            (num_vars) * (num_claims - 1) + 1 - (degree_reduction as usize) * (num_claims - 1);

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

                let mut fix_mle = mle.clone();
                {
                    new_chal.into_iter().enumerate().for_each(|(idx, chal)| {
                        fix_mle.fix_variable(idx, chal);
                    });
                    fix_mle.bookkeeping_table[0]
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        debug!("Returning evals:\n{:#?} ", wlx_evals);
        Ok(wlx_evals)
    }

    /// Computes `l(r^{\star})`
    fn compute_aggregated_challenges(
        &self,
        claims: &ClaimGroup<F>,
        rstar: F,
    ) -> Result<Vec<F>, ClaimError> {
        let claim_vecs = claims.get_claim_points_matrix();

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

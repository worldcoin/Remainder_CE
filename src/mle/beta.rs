use std::{
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};


use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{log2, cfg_into_iter};
use derive_more::{From, Into};
use itertools::{repeat_n, Itertools};
use rayon::{prelude::ParallelIterator, slice::ParallelSlice};

use crate::FieldExt;
use crate::layer::Claim;

use super::{Mle, MleIndex, MleRef, MleAble, MleError};
use thiserror::Error;

pub struct BetaTable<F: FieldExt> {
    layer_claim: Claim<F>,
    table: Vec<F>,
    relevant_indices: Vec<usize>,
}

impl BetaTable for BetaTable<F: FieldExt> {

    fn new(&layer_claim: Claim<F>, mlerefs: &[impl MleRef]) {
        let (layer_claims_idx, _) = layer_claim;
        let relevant_claims: Vec<(F, F)> = mle_refs.iter()
        .map(|mleref| {
        mleref.get_mle_indices().iter().filter(
            |mleindex| matches!(**mleindex, MleIndex::IndexedBit(_))
        ).map(|index| {
            if let MleIndex::IndexedBit(num) = index {
                (F::one() - layer_claims_idx[*num], layer_claims_idx[*num])
            } else {
                (F::one(), F::one())
            }})
            }).flatten()
            .collect();
        let unique_claims: Vec<(F, F)> = relevant_claims.iter().unique().map(|item| *item).collect();
        
        // construct the table, thaler 13
        // TODO (vishruti): make this parallelizable 
        let (one_minus_r, r) = relevafnt_claims[0];
        let mut cur_table = vec![one_minus_r, r];
        for i in 1..relevant_claims.len() {
            let (one_minus_r, r) = relevant_claims[i];
            let mut firsthalf: Vec<F> = cur_table.iter().map(|eval| *eval * one_minus_r).collect();
            let secondhalf: Vec<F> = cur_table.iter().map(|eval| *eval * r).collect();
            firsthalf.extend(secondhalf.iter());
            cur_table = firsthalf;
        }

        BetaTable { layer_claim: layer_claim, table: cur_table }   
    }

    fn beta_update<F: FieldExt>(
        mle: &mut DenseMleRef<F>,
        round_index: usize,
        challenge: F,
    ) -> Result<(), BetaError> {
    
        let (layer_claims, _) = mle.layer_claims.as_ref().ok_or(BetaError::NoBetaTable)?;
        let curr_beta = &mle.beta_table.as_ref().ok_or(BetaError::NoBetaTable);
        let layer_claim = layer_claims[round_index];
    
        // claim can never be 0, otherwise there is no inverse
        let layer_claim_inv = layer_claim.inverse().ok_or(BetaError::NoInverse)?;
    
        // update the beta table given round random challenge, thaler 13
        let mult_factor = layer_claim_inv * (challenge*layer_claim + (F::one()-challenge)*(F::one()-layer_claim));
        let new_beta: Vec<F> = curr_beta.clone().unwrap().iter().skip(1).step_by(2).map(|x| *x*mult_factor).collect();
        mle.beta_table = Some(new_beta);
        Ok(())
    
    }
    
}
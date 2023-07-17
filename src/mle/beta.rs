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

use super::{Mle, MleIndex, MleRef, MleAble};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub struct BetaTable<F: FieldExt> {
    layer_claim: Claim<F>,
    pub table: Vec<F>,
    relevant_indices: Vec<usize>,
}

#[derive(Error, Debug, Clone)]
pub enum BetaError {
    #[error("claim index is 0, cannot take inverse")]
    NoInverse,
    #[error("not enough claims to compute beta table")]
    NotEnoughClaims,
}


impl<F: FieldExt> BetaTable<F> {

    pub fn new(layer_claim: Claim<F>, mlerefs: &[impl MleRef]) -> BetaTable<F> {
        let (layer_claims_idx, _) = &layer_claim;
        let (relevant_idx, relevant_claims): (Vec<usize>, Vec<(F, F)>) = mlerefs.iter()
        .map(|mleref| {
        mleref.get_mle_indices().iter().filter(
            |mleindex| matches!(**mleindex, MleIndex::IndexedBit(_))
        ).map(|index| {
            if let MleIndex::IndexedBit(num) = index {
                (*num, (F::one() - layer_claims_idx[*num], layer_claims_idx[*num]))
            } else {
                (0, (F::one(), F::one()))
            }})
            }).flatten()
            .unzip();
        let unique_idx: Vec<usize> = relevant_idx.iter().unique().map(|idx| *idx).collect();
        let unique_claims: Vec<(F, F)> = relevant_claims.iter().unique().map(|claim| *claim).collect();
        
        // construct the table, thaler 13
        // TODO (vishruti): make this parallelizable 
        let (one_minus_r, r) = unique_claims[0];
        let mut cur_table = vec![one_minus_r, r];
        for i in 1..unique_claims.len() {
            let (one_minus_r, r) = unique_claims[i];
            let mut firsthalf: Vec<F> = cur_table.iter().map(|eval| *eval * one_minus_r).collect();
            let secondhalf: Vec<F> = cur_table.iter().map(|eval| *eval * r).collect();
            firsthalf.extend(secondhalf.iter());
            cur_table = firsthalf;
        }

        BetaTable { layer_claim: layer_claim, table: cur_table, relevant_indices: unique_idx }   
    }

    pub fn beta_update(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<(), BetaError> {
    
        let (layer_claims, _) = &self.layer_claim;
        let curr_beta = &self.table;
        if self.relevant_indices.contains(&round_index) {
            let layer_claim = layer_claims[round_index];
    
            // claim can never be 0, otherwise there is no inverse
            let layer_claim_inv = layer_claim.inverse().ok_or(BetaError::NoInverse)?;
        
            // update the beta table given round random challenge, thaler 13
            let mult_factor = layer_claim_inv * (challenge*layer_claim + (F::one()-challenge)*(F::one()-layer_claim));
            let new_beta: Vec<F> = curr_beta.clone().iter().skip(1).step_by(2).map(|x| *x*mult_factor).collect();
            self.table = new_beta;
        }
        Ok(())
    }
    
}
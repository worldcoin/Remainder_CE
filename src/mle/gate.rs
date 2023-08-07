use std::{
    cmp::max,
    fmt::Debug,
    iter::{Cloned, Zip},
    marker::PhantomData,
};

use ark_ff::BigInt;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter, rand::Rng};
use itertools::Itertools;
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::{layer::Claim, mle::beta::BetaTable, expression::{ExpressionStandard, Expression}};
use crate::FieldExt;

use super::{Mle, MleIndex, MleRef, MleAble, dense::{DenseMleRef, DenseMle}};
use thiserror::Error;

#[derive(Error, Debug, Clone)]

pub struct AddGate<F: FieldExt> {
    layer_claim: Claim<F>,
    nonzero_gates: Vec<((F, F, F), F)>,
    lhs: DenseMleRef<F>,
    rhs: DenseMleRef<F>,
    beta_g: Option<BetaTable<F>>,
}

/// Error handling for beta table construction
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("after phase 1, the lhs should be fully fixed")]
    Phase1Error,
}

impl<F: FieldExt> AddGate<F> {
    /// initialize bookkeeping tables for phase 1 of sumcheck
    pub fn init_phase_1(&mut self) -> DenseMleRef<F> {
        // TODO!(vishady) so many clones
        let mut beta_g = self.beta_g.clone();
        if beta_g.is_none() { 
            beta_g = Some(BetaTable::new(self.layer_claim.clone()).unwrap());
            self.beta_g = beta_g.clone();
        } else {
            beta_g = self.beta_g.clone();
        }
        let num_x = self.lhs.num_vars();
        let mut a_hg = vec![F::zero(); 1 << num_x];
        let _ = self.nonzero_gates.clone().into_iter().map(
            |((z, x, y), val)| {
                let x_ind = x.into_bigint().as_ref()[0] as usize;
                let y_ind = y.into_bigint().as_ref()[0] as usize;
                let z_ind = z.into_bigint().as_ref()[0] as usize;
                let adder = val * beta_g.as_ref().unwrap().table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                a_hg[x_ind] = a_hg[x_ind] + adder + (adder * self.rhs.bookkeeping_table().get(y_ind).unwrap_or(&F::zero()));
            }
        );
        DenseMle::new(a_hg).mle_ref()
    }

    /// initialize bookkeeping tables for phase 2 of sumcheck
    pub fn init_phase_2(&self, phase_1_claim_chal: Vec<F>, f_at_u: F) -> (DenseMleRef<F>, [DenseMleRef<F>; 2]){
        let beta_g = self.beta_g.as_ref().expect("beta table should be initialized by now");
        // uhhhhhh
        let phase_1_claim = (phase_1_claim_chal, F::zero()); 
        let beta_u = BetaTable::new(phase_1_claim).unwrap();
        let num_y = self.rhs.num_vars();
        let mut a_f1_lhs = vec![F::zero(); 1 << num_y];
        let mut a_f1_rhs = vec![F::zero(); 1 << num_y];
        let _ = self.nonzero_gates.clone().into_iter().map(
            |((z, x, y), val)| {
                let x_ind = x.into_bigint().as_ref()[0] as usize;
                let y_ind = y.into_bigint().as_ref()[0] as usize;
                let z_ind = z.into_bigint().as_ref()[0] as usize;
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let adder = gz * ux * val;
                a_f1_lhs[y_ind] = a_f1_lhs[y_ind] + adder * f_at_u;
                a_f1_rhs[y_ind] = a_f1_rhs[y_ind] + adder;
            }
        );
        (DenseMle::new(a_f1_lhs).mle_ref(), [DenseMle::new(a_f1_rhs).mle_ref(), self.rhs.clone()])
    }

    pub fn sumcheck(&mut self) -> Result<(DenseMleRef<F>, DenseMleRef<F>), GateError> {
        // do first (num_x_vars) rounds of sumcheck

        // do the next (num_y_vars) rounds of sumcheck
        
        todo!()
    }

}


#[cfg(test)]
mod test {
    use crate::mle::{dense::DenseMle, Mle};

    use super::*;
    use ark_bn254::Fr;
    use ark_std::One;

    #[test]
    fn test_expected_val() {

    }
}
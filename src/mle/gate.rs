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

use crate::{layer::Claim, mle::beta::BetaTable, expression::{ExpressionStandard, Expression}, sumcheck::*};
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

/// Error handling for gate mle construction
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("after phase 1, the lhs should be fully fixed")]
    Phase1Error,
    #[error("mle not fully bound")]
    MleNotFullyBound,
}

/// shut up
pub fn evaluate_mle_ref_product_gate<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    independent_variable: bool,
    degree: usize,
) -> Result<PartialSum<F>, MleError> {
    for mle_ref in mle_refs {
        if !mle_ref.indexed() {
            return Err(MleError::NotIndexedError);
        }
    }
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    // let real_num_vars = if independent_variable {
    //     max_num_vars
    // } else {
    //     max_num_vars
    // };

    let real_num_vars = if max_num_vars != 0 {
        max_num_vars - 1
    } else {
        max_num_vars
    };

    if independent_variable {
        //There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
        let eval_count = degree + 1;

        //iterate across all pairs of evaluations
        let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
            #[cfg(feature = "parallel")]
            || vec![F::zero(); eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::zero(); eval_count],
            |mut acc, index| {
                
                //get the product of all evaluations over 0/1/..degree
                let evals = mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let zero = F::zero();
                        let index = if mle_ref.num_vars() < max_num_vars {
                            let max = 1 << mle_ref.num_vars();
                            (index * 2) % max
                        } else {
                            index * 2
                        };
                        let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);
                        let second = if mle_ref.num_vars() != 0 {
                            *mle_ref.bookkeeping_table().get(index + 1).unwrap_or(&zero)
                        } else {
                            first
                        };

                        // let second = *mle_ref.mle().get(index + 1).unwrap_or(&zero);
                        let step = second - first;

                        let successors =
                            std::iter::successors(Some(second), move |item| Some(*item + step));
                        //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
                        std::iter::once(first).chain(successors)
                    })
                    .map(|item| -> Box<dyn Iterator<Item = F>> { Box::new(item) })
                    .reduce(|acc, evals| Box::new(acc.zip(evals).map(|(acc, eval)| acc * eval)))
                    .unwrap();

                acc.iter_mut()
                    .zip(evals)
                    .for_each(|(acc, eval)| *acc += eval);
                acc
            },
        );

        #[cfg(feature = "parallel")]
        let evals = evals.reduce(
            || vec![F::zero(); eval_count],
            |mut acc, partial| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );

        //     dbg!(&evals, real_num_vars);
        Ok(PartialSum {
            sum_or_eval: SumOrEvals::Evals(evals),
            max_num_vars: real_num_vars,
        })
    } else {
        // There is no independent variable and we can sum over everything
        let partials = cfg_into_iter!((0..1 << (max_num_vars))).fold(
            #[cfg(feature = "parallel")]
            || F::zero(),
            #[cfg(not(feature = "parallel"))]
            F::zero(),
            |acc, index| {
                // Go through each MLE within the product
                let product = mle_refs
                    .iter()
                    // Result of this `map()`: A list of evaluations of the MLEs at `index`
                    .map(|mle_ref| {
                        let index = if mle_ref.num_vars() < max_num_vars {
                            // max = 2^{num_vars}; index := index % 2^{num_vars}
                            let max = 1 << mle_ref.num_vars();
                            index % max
                        } else {
                            index
                        };
                        // --- Access the MLE at that index. Pad with zeros ---
                        mle_ref.bookkeeping_table().get(index).cloned().unwrap_or(F::zero())
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                // --- Combine them into the accumulator ---
                // Note that the accumulator stores g(0), g(1), ..., g(d - 1)
                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = partials.sum();
        Ok(PartialSum {
            sum_or_eval: SumOrEvals::Sum(sum),
            max_num_vars: real_num_vars,
        })
    }
}



/// shut up
pub fn evaluate_mle_ref_product_gate_fullybound<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    challenges: Vec<F>,
) -> Result<F, GateError> {
    challenges
            .into_iter()
            .enumerate()
            .for_each(|(round_idx, challenge)| {
                mle_refs.iter_mut().for_each(|mle_ref| {
                    mle_ref.fix_variable(round_idx, challenge);
                });
            });
    mle_refs.into_iter().fold(Ok(F::one()), |acc, new_mle_ref| {
    // --- Accumulate either errors or multiply ---
    if let Err(e) = acc {
        return Err(e);
    }
    if new_mle_ref.bookkeeping_table().len() != 1 {
        return Err(GateError::MleNotFullyBound);
    }
    Ok(acc.unwrap() * new_mle_ref.bookkeeping_table()[0])
    })
}

impl<F: FieldExt> AddGate<F> {

    /// new addgate mle 
    pub fn new(layer_claim: Claim<F>, nonzero_gates: Vec<((F, F, F), F)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>) -> AddGate<F> {
        AddGate {
            layer_claim: layer_claim,
            nonzero_gates: nonzero_gates,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            beta_g: None,
        }
    }

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

        self.nonzero_gates.clone().into_iter().for_each(
            |((z, x, y), val)| {
                let x_ind = x.into_bigint().as_ref()[0] as usize;
                let y_ind = y.into_bigint().as_ref()[0] as usize;
                let z_ind = z.into_bigint().as_ref()[0] as usize;
                let adder = val * beta_g.as_ref().unwrap().table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                a_hg[x_ind] = a_hg[x_ind] + adder + (adder * self.rhs.bookkeeping_table().get(y_ind).unwrap_or(&F::zero()));
            }
        );
        dbg!(&a_hg);
        DenseMle::new(a_hg).mle_ref()
    }

    /// initialize bookkeeping tables for phase 2 of sumcheck
    pub fn init_phase_2(&self, u_claim: Claim<F>) -> (DenseMleRef<F>, [DenseMleRef<F>; 2]) {
        let (_, f_at_u) = u_claim;
        let beta_g = self.beta_g.as_ref().expect("beta table should be initialized by now");
        let beta_u = BetaTable::new(u_claim).unwrap();
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

    /// shut up
    pub fn sumcheck_phase_1(&mut self, rng: &mut impl Rng) -> Vec<(Vec<F>, Option<F>)> {

        // initialization
        let mut phase1_mle = self.init_phase_1();
        phase1_mle.index_mle_indices(0);
        let mut lhs = self.lhs.clone();
        lhs.index_mle_indices(0);
        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenge: Option<F> = None;
        let num_rounds_phase1 = self.lhs.num_vars();
        
        // sumcheck rounds (binding x)
        for round in 0..num_rounds_phase1 {
            dbg!(round);
            if let Some(challenge) = challenge {
                phase1_mle.fix_variable(round - 1, challenge);
                //dbg!(&phase1_mle, round);
                lhs.fix_variable(round - 1, challenge);
                
            }
            //dbg!(&phase1_mle, round);
            let degree = 1;
            let independent_variable = phase1_mle.mle_indices().contains(&MleIndex::IndexedBit(round));
            let eval = evaluate_mle_ref_product_gate(&[phase1_mle.clone()], independent_variable, degree).unwrap();

    
            if let PartialSum { sum_or_eval: SumOrEvals::Evals(evaluations), max_num_vars: _ } = eval {
                messages.push((evaluations, challenge));
            } else {
                panic!();
            };
    
            challenge = Some(F::rand(rng));
            //challenge = Some(F::from(2_u64));
        }
        
        messages
    }


    /// shut up
    pub fn sumcheck_phase_2(&mut self, rng: &mut impl Rng, challenges: Vec<F>, claimed_val: F) -> Vec<(Vec<F>, Option<F>)> {

        // init
        let (mut phase2_lhs, mut phase2_rhs) = self.init_phase_2((challenges, claimed_val));
        let num_rounds_phase2 = self.rhs.num_vars();
        phase2_lhs.index_mle_indices(0);
        phase2_rhs
                .iter_mut()
                .for_each(|mle_ref| {
                    mle_ref.index_mle_indices(0);
                });
        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenge: Option<F> = None;

        // sumcheck rounds (binding y)
        for round in 0..num_rounds_phase2 {
            dbg!(round);
            if let Some(challenge) = challenge {
                phase2_lhs.fix_variable(round-1, challenge);
                phase2_rhs
                .iter_mut()
                .for_each(|mle_ref| {
                    mle_ref.fix_variable(round-1, challenge); 
            });
            }
    
            let degree = 2;
            let independent_variable_lhs = phase2_lhs.mle_indices().contains(&MleIndex::IndexedBit(round));
            let eval_lhs = evaluate_mle_ref_product_gate(&[phase2_lhs.clone()], independent_variable_lhs, degree).unwrap();

            let independent_variable_rhs = phase2_rhs[0].mle_indices().contains(&MleIndex::IndexedBit(round)) | phase2_rhs[1].mle_indices().contains(&MleIndex::IndexedBit(round));
            let eval_rhs = evaluate_mle_ref_product_gate(&phase2_rhs.clone(), independent_variable_rhs, degree).unwrap();

            let eval = eval_lhs + eval_rhs;

            if let PartialSum { sum_or_eval: SumOrEvals::Evals(evaluations), max_num_vars: _ } = eval {
                messages.push((evaluations, challenge));
            } else {
                panic!();
            };
    
            challenge = Some(F::rand(rng));
        }
        
        messages
    }


    /// verifier for gate sumcheck phase 1
    pub fn gate_verify_1(
        &self,
        messages: Vec<(Vec<F>, Option<F>)>,
        phase_1_mle: DenseMleRef<F>,
        rng: &mut impl Rng,
    ) -> Result<Claim<F>, VerifyError> {
        let mut prev_evals = &messages[0].0;
        let mut chal = F::zero();
    
        let mut challenges = vec![];
        let num_lhs = self.lhs.num_vars();
        dbg!(num_lhs);

        //TODO: first round here 
    
        // --- Go through sumcheck messages + (FS-generated) challenges ---
        // Round j, 1 < j < u for the first u rounds of sumcheck
        for i in 1..(num_lhs) {
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");
    
            dbg!(prev_at_r);
            dbg!(curr_evals[0] + curr_evals[1]);
            dbg!(i);
            dbg!("lhs");
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);
        }
    
        // final round for binding x
        let final_chal = F::rand(rng);
        challenges.push(final_chal);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).expect("could not evaluate at challenge point");
        let oracle_query = evaluate_mle_ref_product_gate_fullybound(&mut [phase_1_mle], challenges.clone()).unwrap();
        if prev_at_r != oracle_query {
            return Err(VerifyError::SumcheckBad);
        };
        Ok((challenges, oracle_query))
    }

    /// verifier for gate sumcheck phase 1
    pub fn gate_verify_2(
        &self,
        messages: Vec<(Vec<F>, Option<F>)>,
        phase_2_lhs: DenseMleRef<F>,
        phase_2_rhs: &mut [impl MleRef<F = F>],
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Result<Claim<F>, VerifyError> {
        let mut prev_evals = &messages[0].0;
        let mut chal = F::zero();
        let claimed_val = messages[0].0[0] + messages[0].0[1];
        if claimed_val != claim.1 {
            return Err(VerifyError::SumcheckBad);
        }
    
        let mut challenges = vec![];
        let num_rhs = self.rhs.num_vars();
        dbg!(num_rhs);
    
        // --- Go through sumcheck messages + (FS-generated) challenges ---
        // Round k, 1 < k < v for the next v rounds of sumcheck
        for i in 1..(num_rhs) {
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");
    
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);
        }
    
        // final round for binding y
        let final_chal = F::rand(rng);
        challenges.push(final_chal);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).expect("could not evaluate at challenge point");
        let oracle_query_lhs = evaluate_mle_ref_product_gate_fullybound(&mut [phase_2_lhs], challenges.clone()).unwrap();
        let oracle_query_rhs = evaluate_mle_ref_product_gate_fullybound(phase_2_rhs, challenges.clone()).unwrap();
        let oracle_query = oracle_query_lhs + oracle_query_rhs;
        if prev_at_r != oracle_query {
            return Err(VerifyError::SumcheckBad);
        };
        Ok((challenges, oracle_query))
    }

    fn gate_verify(
        &self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
    ) -> () {

        todo!()
    }

}



#[cfg(test)]
mod test {
    use crate::mle::{dense::DenseMle, Mle};

    use super::*;
    use ark_bn254::Fr;
    use ark_std::One;
    use ark_std::test_rng;

    #[test]
    fn test_sumcheck_1() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(1),Fr::from(3),Fr::from(2),], Fr::one());
        let nonzero_gates = vec![
            ((Fr::from(1), Fr::from(1), Fr::from(1)), Fr::from(1)),
            ((Fr::from(3), Fr::from(0), Fr::from(2)), Fr::from(1)),
            ((Fr::from(2), Fr::from(3), Fr::from(4)), Fr::from(1)),
            ];

        let lhs_v = vec![
            Fr::from(12980911),
            Fr::from(118408),
            Fr::from(1108),
            Fr::from(10180910),
        ];
        let lhs_mle_ref = DenseMle::new(lhs_v).mle_ref();

        let rhs_v = vec![
            Fr::from(1498),
            Fr::from(2408291),
            Fr::from(14029189),
            Fr::from(1287891),
        ];
        let rhs_mle_ref = DenseMle::new(rhs_v).mle_ref();

        let mut gate_mle = AddGate::new(claim.clone(), nonzero_gates, lhs_mle_ref, rhs_mle_ref);
        let messages = gate_mle.sumcheck(&mut rng);
        let verify_res = gate_mle.gate_verify(messages);
        assert!(verify_res.is_ok());

    }

    #[test]
    fn test_sumcheck_2() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(1),Fr::from(3),Fr::from(2),], Fr::one());
        let nonzero_gates = vec![
            ((Fr::from(1), Fr::from(1), Fr::from(1)), Fr::from(1)),
            ((Fr::from(3), Fr::from(0), Fr::from(2)), Fr::from(1)),
            ((Fr::from(2), Fr::from(3), Fr::from(4)), Fr::from(1)),
            ];

        let lhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
        ];
        let lhs_mle_ref = DenseMle::new(lhs_v).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
        ];
        let rhs_mle_ref = DenseMle::new(rhs_v).mle_ref();

        let mut gate_mle = AddGate::new(claim.clone(), nonzero_gates, lhs_mle_ref, rhs_mle_ref);
        let messages = gate_mle.sumcheck(&mut rng);
        let verify_res = gate_mle.gate_verify(messages);
        assert!(verify_res.is_ok());

    }
}
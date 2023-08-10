use std::{
    fmt::Debug,
    marker::PhantomData,
};
use ark_std::{cfg_into_iter, rand::Rng};
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{layer::{Claim, Layer, LayerError, LayerId, LayerBuilder, VerificationError, claims::ClaimError}, mle::beta::BetaTable, expression::ExpressionStandard, sumcheck::*, transcript::Transcript, prover::SumcheckProof};
use crate::FieldExt;

use super::{MleIndex, MleRef, dense::{DenseMleRef, DenseMle}};
use thiserror::Error;


impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for AddGate<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(&mut self, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<SumcheckProof<F>, LayerError> {
        // initialization
        let first_message = self.init_phase_1(claim).expect("could not evaluate original lhs and rhs");
        let (phase_1_lhs, phase_1_rhs) = self.phase_1_mles.as_mut().ok_or(GateError::Phase1InitError).unwrap();

        let mut challenges: Vec<F> = vec![];
        transcript
        .append_field_elements("Initial Sumcheck evaluations", &first_message)
        .unwrap();
        let num_rounds_phase1 = self.lhs.num_vars();
        
        // sumcheck rounds (binding x)
        let mut sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
        .chain((1..num_rounds_phase1).map(|round| {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
            challenges.push(challenge);
            let eval = prove_round(round, challenge, phase_1_lhs, phase_1_rhs).unwrap();
            transcript
                .append_field_elements("Sumcheck evaluations", &eval)
                .unwrap();
            Ok::<_, LayerError>(eval)
        }))
        .try_collect()?;
        

        let final_chal_u = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal_u);
        let last_lhs = check_fully_bound(phase_1_lhs, challenges.clone()).unwrap();
        let last_rhs = check_fully_bound(phase_1_rhs, challenges.clone()).unwrap();

        let f_2 = &phase_1_lhs[1];
        if f_2.bookkeeping_table.len() == 1 {
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = (challenges.clone(), last_lhs + last_rhs);
        
            let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();
            let (phase_2_lhs, phase_2_rhs) = self.phase_2_mles.as_mut().ok_or(GateError::Phase2InitError).unwrap();
            
            transcript
            .append_field_elements("Initial Sumcheck evaluations", &first_message)
            .unwrap();
            let num_rounds_phase2 = self.rhs.num_vars();


            let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_phase2).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                challenges.push(challenge);
                let eval = prove_round(round, challenge, phase_2_lhs, phase_2_rhs).unwrap();
                transcript
                    .append_field_elements("Sumcheck evaluations", &eval)
                    .unwrap();
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

            sumcheck_rounds.extend(sumcheck_rounds_y.into_iter());
            // sumcheck rounds (binding y)

            Ok(sumcheck_rounds.into())
        }
        else {
            return Err(LayerError::LayerNotReady);
        }
        
    }


    /// Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError> {

        let mut prev_evals = &sumcheck_rounds[0];
    
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let num_u = self.lhs.num_vars();

        //TODO: first round here 
        let claimed_claim = prev_evals[0] + prev_evals[1];
        if claimed_claim != claim.1 {
            return Err(LayerError::VerificationError(VerificationError::SumcheckStartFailed));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();
    

        for (i, curr_evals) in sumcheck_rounds.iter().enumerate().skip(1) {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
    
                let prev_at_r = evaluate_at_a_point(prev_evals, challenge)
                    .map_err(|err| LayerError::InterpError(err))?;
    
                if prev_at_r != curr_evals[0] + curr_evals[1] {
                    return Err(LayerError::VerificationError(
                        VerificationError::SumcheckFailed,
                    ));
                };
    
                transcript
                    .append_field_elements("Sumcheck evaluations", &curr_evals)
                    .unwrap();
    
                prev_evals = curr_evals;
                challenges.push(challenge);
                if (1..(num_u+1)).contains(&i) {
                    first_u_challenges.push(challenge);
                }
                else {
                    last_v_challenges.push(challenge);
                }
            }

        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        last_v_challenges.push(final_chal);
        let (phase_1_lhs, phase_1_rhs) = self.phase_1_mles.as_mut().unwrap();
        let (phase_2_lhs, phase_2_rhs) = self.phase_2_mles.as_mut().unwrap();
        let first_lhs = check_fully_bound(phase_1_lhs, last_v_challenges.clone());
        let first_rhs = check_fully_bound(phase_1_rhs, last_v_challenges.clone());
        
        if let (Err(_), _) = (&first_lhs, &first_rhs) {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }
        else if let (_, Err(_)) = (&first_lhs, &first_rhs) {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }
        else {
            let last_lhs = check_fully_bound(phase_2_lhs, last_v_challenges.clone()).unwrap();
            let last_rhs = check_fully_bound(phase_2_rhs, last_v_challenges.clone()).unwrap();
            let oracle_query = last_lhs + last_rhs;
            let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

            if oracle_query != prev_at_r {
                return Err(LayerError::VerificationError(
                    VerificationError::FinalSumcheckFailed,
                ));
            }
        }
        
        Ok(())
    }

    ///Get the claims that this layer makes on other layers
    fn claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        let mut claims: Vec<(LayerId, Claim<F>)> = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];
        if let Some(([_, f_2_u], _)) = &self.phase_1_mles {
            for index in f_2_u.mle_indices() {
                fixed_mle_indices_u.push(index.val().ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?);
            }
            let val = f_2_u.bookkeeping_table()[0];
            claims.push((self.id().clone(), (fixed_mle_indices_u, val)));
        }
        else {
            return Err(LayerError::LayerNotReady)
        }
        let mut fixed_mle_indices_v: Vec<F> = vec![];
        if let Some((_, [_, f_3_v])) = &self.phase_2_mles {
            for index in f_3_v.mle_indices() {
                fixed_mle_indices_v.push(index.val().ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?);
            }
            let val = f_3_v.bookkeeping_table()[0];
            claims.push((self.id().clone(), (fixed_mle_indices_v, val)));
        }
        else {
            return Err(LayerError::LayerNotReady)
        }
        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Gets this layers expression
    fn expression(&self) -> &ExpressionStandard<F> {
        // uh this is def not right anyway we dont need it??? we're not using expressions
        return &self.expression
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        todo!()
    }
}

/// Error handling for gate mle construction
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("phase 1 not initialized")]
    Phase1InitError,
    #[error("phase 2 not initialized")]
    Phase2InitError,
    #[error("mle not fully bound")]
    MleNotFullyBoundError,
    #[error("failed to bind variables during sumcheck")]
    SumcheckProverError,
    #[error("last round sumcheck fail")]
    SumcheckFinalFail,
    #[error("empty list for lhs or rhs")]
    EmptyMleList,
    #[error("bound indices fail to match challenge")]
    EvaluateBoundIndicesDontMatch,
}


#[derive(Error, Debug)]
/// very (not) cool addgate 
pub struct AddGate<F: FieldExt, Tr: Transcript<F>> {
    layer_id: LayerId,
    nonzero_gates: Vec<(usize, usize, usize)>,
    lhs: DenseMleRef<F>,
    rhs: DenseMleRef<F>,
    expression: ExpressionStandard<F>,
    beta_g: Option<BetaTable<F>>,
    phase_1_mles: Option<([DenseMleRef<F>; 2], [DenseMleRef<F>; 1])>,
    phase_2_mles: Option<([DenseMleRef<F>; 1], [DenseMleRef<F>; 2])>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> AddGate<F, Tr> {

    /// new addgate mle 
    pub fn new(layer_id: LayerId, nonzero_gates: Vec<(usize, usize, usize)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>) -> AddGate<F, Tr> {
        AddGate {
            layer_id: layer_id,
            nonzero_gates: nonzero_gates,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            // this makes NO sense please dont think anything of this
            expression: ExpressionStandard::Sum(Box::new(ExpressionStandard::Mle(lhs.clone())), Box::new(ExpressionStandard::Mle(rhs.clone()))),
            beta_g: None,
            phase_1_mles: None,
            phase_2_mles: None,
            _marker: PhantomData,
        }
    }

    fn set_beta_g(&mut self, betag: BetaTable<F>) {
        self.beta_g = Some(betag);
    }

    fn set_phase_1(&mut self, (mle_l, mle_r): ([DenseMleRef<F>; 2], [DenseMleRef<F>; 1])) {
        self.phase_1_mles = Some((mle_l, mle_r));
    }

    fn set_phase_2(&mut self, (mle_l, mle_r): ([DenseMleRef<F>; 1], [DenseMleRef<F>; 2])) {
        self.phase_2_mles = Some((mle_l, mle_r));
    }

    /// initialize bookkeeping tables for phase 1 of sumcheck
    pub fn init_phase_1(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        // TODO!(vishady) so many clones
        let beta_g = BetaTable::new(claim).unwrap();
        self.set_beta_g(beta_g.clone());
        self.lhs.index_mle_indices(0);
        let num_x = self.lhs.num_vars();
        let mut a_hg_lhs = vec![F::zero(); 1 << num_x];
        let mut a_hg_rhs = vec![F::zero(); 1 << num_x];

        self.nonzero_gates.clone().into_iter().for_each(
            |(z_ind, x_ind, y_ind)| {
                let adder = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                a_hg_lhs[x_ind] = a_hg_lhs[x_ind] + adder;
                a_hg_rhs[x_ind] = a_hg_rhs[x_ind] + (adder * *self.rhs.bookkeeping_table().get(y_ind).unwrap_or(&F::zero()));
            }
        );

        let mut phase_1_lhs = [DenseMle::new(a_hg_lhs).mle_ref(), self.lhs.clone()];
        let mut phase_1_rhs = [DenseMle::new(a_hg_rhs).mle_ref()];
        index_mle_indices_gate(phase_1_lhs.as_mut(), 0);
        index_mle_indices_gate(phase_1_rhs.as_mut(), 0);
        self.set_phase_1((phase_1_lhs.clone(), phase_1_rhs.clone()));
        compute_sumcheck_message_gate(&phase_1_lhs, &phase_1_rhs, 0)

        
    }

    /// initialize bookkeeping tables for phase 2 of sumcheck
    pub fn init_phase_2(&mut self, u_claim: Claim<F>, f_at_u: F) -> Result<Vec<F>, GateError> {
        let beta_g = self.beta_g.as_ref().expect("beta table should be initialized by now");
        let beta_u = BetaTable::new(u_claim).unwrap();
        let num_y = self.rhs.num_vars();
        let mut a_f1_lhs = vec![F::zero(); 1 << num_y];
        let mut a_f1_rhs = vec![F::zero(); 1 << num_y];
        self.nonzero_gates.clone().into_iter().for_each(
            |(z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let adder = gz * ux;
                a_f1_lhs[y_ind] = (a_f1_lhs[y_ind] + adder) * f_at_u;
                a_f1_rhs[y_ind] = a_f1_rhs[y_ind] + adder;
            }
        );
        let mut phase_2_lhs = [DenseMle::new(a_f1_lhs).mle_ref()];
        let mut phase_2_rhs = [DenseMle::new(a_f1_rhs).mle_ref(), self.rhs.clone()];
        index_mle_indices_gate(phase_2_lhs.as_mut(), 0);
        index_mle_indices_gate(phase_2_rhs.as_mut(), 0);
        self.set_phase_2((phase_2_lhs.clone(), phase_2_rhs.clone()));
        compute_sumcheck_message_gate(&phase_2_lhs, &phase_2_rhs, 0)
    }

    /// shut up
    pub fn dummy_prove_rounds(&mut self, claim: Claim<F>, rng: &mut impl Rng) -> Vec<(Vec<F>, Option<F>)> {

        // initialization
        let first_message = self.init_phase_1(claim).expect("could not evaluate original lhs and rhs");
        let (phase_1_lhs, phase_1_rhs) = self.phase_1_mles.as_mut().ok_or(GateError::Phase1InitError).unwrap();

        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        let num_rounds_phase1 = self.lhs.num_vars();
        
        // sumcheck rounds (binding x)
        for round in 1..(num_rounds_phase1) {
            challenge = Some(F::rand(rng));
            let chal = challenge.unwrap();
            challenges.push(chal);
            let eval = prove_round(round, chal, phase_1_lhs, phase_1_rhs).unwrap();
            messages.push((eval, challenge));
        }

        let final_chal = Some(F::rand(rng)).unwrap();
        challenges.push(final_chal);
        let last_lhs = check_fully_bound(phase_1_lhs, challenges.clone()).unwrap();
        let last_rhs = check_fully_bound(phase_1_rhs, challenges.clone()).unwrap();
        let lhs = self.lhs.clone();
        let f_2 = &phase_1_lhs[1];
        if f_2.bookkeeping_table.len() == 1 {
        }
        else {

        }
        let f_at_u = check_fully_bound(&mut [lhs], challenges.clone()).unwrap();
        let u_challenges = (challenges.clone(), last_lhs + last_rhs);
        

        let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();
        let (phase_2_lhs, phase_2_rhs) = self.phase_2_mles.as_mut().ok_or(GateError::Phase2InitError).unwrap();
        messages.push((first_message, Some(final_chal)));
        let num_rounds_phase2 = self.rhs.num_vars();

        // sumcheck rounds (binding y)
        for round in 1..num_rounds_phase2 {
            challenge = Some(F::rand(rng));
            let chal = challenge.unwrap();
            challenges.push(chal);
            let eval = prove_round(round, chal, phase_2_lhs, phase_2_rhs).unwrap();
            messages.push((eval, challenge));
        }
        
        messages
    }

    /// verifier for gate sumcheck phase 1
    pub fn dummy_verify_rounds(
        &mut self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
        claim: Claim<F>,
    ) -> Result<(), VerifyError> {
        let mut prev_evals = &messages[0].0;
    
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let num_u = self.lhs.num_vars();
        let num_v = self.rhs.num_vars();
        let num_rounds = num_u + num_v;

        //TODO: first round here 
        let claimed_val = messages[0].0[0] + messages[0].0[1];
        if claimed_val != claim.1 {
            return Err(VerifyError::SumcheckBad);
        }
    
        // --- Go through sumcheck messages + (FS-generated) challenges ---
        // Round j, 1 < j < u for the first u rounds of sumcheck
        for i in 1..(num_rounds) {
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            let chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");
    
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);

            if (1..(num_u+1)).contains(&i) {
                first_u_challenges.push(chal);
            }
            else {
                last_v_challenges.push(chal);
            }
        }

        let final_chal = Some(F::rand(rng)).unwrap();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);

        let mut lhs = self.lhs.clone();
        lhs.index_mle_indices(0);
        let mut rhs = self.rhs.clone();
        rhs.index_mle_indices(0);
        let f_2_u = check_fully_bound(&mut [lhs], first_u_challenges.clone()).unwrap();
        let f_3_v = check_fully_bound(&mut [rhs], last_v_challenges.clone()).unwrap();
        let beta_u = BetaTable::new((first_u_challenges.clone(), f_2_u)).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), f_3_v)).unwrap();
        let beta_g = self.beta_g.as_ref().unwrap();
        let mut f_1_uv = F::zero();
        self.nonzero_gates.clone().into_iter().for_each(
            |(z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                f_1_uv += gz * ux * vy;
            }
        );
        let oracle_query_last_v = f_1_uv * (f_2_u + f_3_v);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        if oracle_query_last_v != prev_at_r {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }

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
pub fn check_fully_bound<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    challenges: Vec<F>,
) -> Result<F, GateError> {
    let mles_bound: Vec<bool> = mle_refs.iter()
        .map(|mle_ref| {
            let indices = mle_ref
            .mle_indices()
            .iter()
            .filter_map(|index| match index {
                MleIndex::Bound(chal, index) => Some((*chal, index)),
                _ => None,
            })
        .collect_vec();

        let start = *indices[0].1;
        let end = *indices[indices.len() - 1].1;

        let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

        if indices.as_slice() != &challenges[start..=end] {
            false
        }
        else {
            true
        }
    }).collect();

    if mles_bound.contains(&false) {
        return Err(GateError::EvaluateBoundIndicesDontMatch);
    }

    mle_refs.into_iter().fold(Ok(F::one()), |acc, mle_ref| {
    // --- Accumulate either errors or multiply ---
    if let Err(e) = acc {
        return Err(e);
    }
    if mle_ref.bookkeeping_table().len() != 1 {
        return Err(GateError::MleNotFullyBoundError);
    }
    Ok(acc.unwrap() * mle_ref.bookkeeping_table()[0])
    })
}

/// shhhh
pub fn index_mle_indices_gate<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    index: usize,
) {
    mle_refs.iter_mut().for_each(
        |mle_ref| {
            mle_ref.index_mle_indices(index);
        }
    )
}

/// bye
pub fn fix_var_gate<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    round_index: usize,
    challenge: F,
) {
    mle_refs.iter_mut().for_each(
        |mle_ref| {
            if mle_ref.mle_indices().contains(&MleIndex::IndexedBit(round_index)) {
                mle_ref.fix_variable(round_index, challenge);
            }
        }
    )
}

/// byebye
fn compute_sumcheck_message_gate<F: FieldExt>(
    lhs: &[impl MleRef<F = F>],
    rhs: &[impl MleRef<F = F>],
    round_index: usize,
) -> Result<Vec<F>, GateError> {

    let degree = 2;
    let independent_variable_lhs = lhs
    .iter()
    .map(|mle_ref| {
        mle_ref
            .mle_indices()
            .contains(&MleIndex::IndexedBit(round_index))
    })
    .reduce(|acc, item| acc | item)
    .ok_or(GateError::EmptyMleList)?;
    let eval_lhs = evaluate_mle_ref_product_gate(lhs, independent_variable_lhs, degree).unwrap();

    let independent_variable_rhs = rhs
    .iter()
    .map(|mle_ref| {
        mle_ref
            .mle_indices()
            .contains(&MleIndex::IndexedBit(round_index))
    })
    .reduce(|acc, item| acc | item)
    .ok_or(GateError::EmptyMleList)?;    let eval_rhs = evaluate_mle_ref_product_gate(rhs, independent_variable_rhs, degree).unwrap();
    let eval = eval_lhs + eval_rhs;


    if let PartialSum { sum_or_eval: SumOrEvals::Evals(evaluations), max_num_vars: _ } = eval {
        Ok(evaluations)
    } else {
        Err(GateError::SumcheckProverError)
    }
}

///Computes a round of the sumcheck protocol on this Layer
fn prove_round<F: FieldExt>(round_index: usize, challenge: F, lhs: &mut [impl MleRef<F = F>], rhs: &mut [impl MleRef<F = F>]) -> Result<Vec<F>, GateError> {
    fix_var_gate(lhs, round_index - 1, challenge);
    fix_var_gate(rhs, round_index - 1, challenge);
    compute_sumcheck_message_gate(lhs, rhs, round_index)
}

#[cfg(test)]
mod test {
    use crate::mle::dense::DenseMle;
    use crate::transcript::poseidon_transcript::PoseidonTranscript;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn test_sumcheck_1() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(1), Fr::from(0), Fr::from(0)], Fr::from(4));
        let nonzero_gates = vec![
            (1, 1, 1),
            ];

        let lhs_v = vec![
            Fr::from(1),
            Fr::from(2),
        ];
        let lhs_mle_ref = DenseMle::new(lhs_v).mle_ref();

        let rhs_v = vec![
            Fr::from(51395810),
            Fr::from(2),
        ];
        let rhs_mle_ref = DenseMle::new(rhs_v).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0), nonzero_gates, lhs_mle_ref, rhs_mle_ref);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    fn test_sumcheck_2() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(0),Fr::from(1), Fr::from(0)], Fr::from(2));
        let nonzero_gates = vec![
            (1, 0, 1),
            (3, 0, 2),
            (2, 3, 4),
            ];

        let lhs_v = vec![
            Fr::from(-19051),
            Fr::from(119084),
            Fr::from(857911),
            Fr::from(1),
            Fr::from(189571),
            Fr::from(16781),
            Fr::from(75361),
            Fr::from(-91901),
        ];
        let lhs_mle_ref = DenseMle::new(lhs_v).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(24251),
            Fr::from(87591),
            Fr::from(1),
            Fr::from(772751),
            Fr::from(-131899),
            Fr::from(191),
        ];
        let rhs_mle_ref = DenseMle::new(rhs_v).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0),  nonzero_gates, lhs_mle_ref, rhs_mle_ref);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());


    }

    #[test]
    fn test_sumcheck_3() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(1),Fr::from(1),Fr::from(0),], Fr::from(5200));
        let nonzero_gates = vec![
            (1, 0, 15),
            (3, 0, 2),
            (5, 3, 14),
            (2, 3, 4),
            ];

        let lhs_v = vec![
            Fr::from(-19051),
            Fr::from(119084),
            Fr::from(857911),
            Fr::from(1),
            Fr::from(189571),
            Fr::from(16781),
            Fr::from(75361),
            Fr::from(-91901),
        ];
        let lhs_mle_ref = DenseMle::new(lhs_v).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(24251),
            Fr::from(87591),
            Fr::from(1),
            Fr::from(772751),
            Fr::from(-131899),
            Fr::from(191),
            Fr::from(80951),
            Fr::from(51),
            Fr::from(-2),
            Fr::from(2),
            Fr::from(1),
            Fr::from(3),
            Fr::from(7),
            Fr::from(9999),
        ];
        let rhs_mle_ref = DenseMle::new(rhs_v).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0), nonzero_gates, lhs_mle_ref, rhs_mle_ref);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }
}
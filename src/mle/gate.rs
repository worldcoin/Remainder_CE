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


/// implement the layer trait for addgate struct
impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for AddGate<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(&mut self, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<SumcheckProof<F>, LayerError> {
        // initialization, get the first sumcheck message
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
            // if there are copy bits, we want to start at that index
            let eval = prove_round(round + self.num_copy_bits, challenge, phase_1_lhs, phase_1_rhs).unwrap();
            transcript
                .append_field_elements("Sumcheck evaluations", &eval)
                .unwrap();
            Ok::<_, LayerError>(eval)
        }))
        .try_collect()?;
        

        // final challenge after binding x (left side of the sum)
        let final_chal_u = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal_u);

        fix_var_gate(phase_1_lhs, num_rounds_phase1 - 1 + self.num_copy_bits, final_chal_u);
        fix_var_gate(phase_1_rhs, num_rounds_phase1 - 1 + self.num_copy_bits, final_chal_u);

        let last_lhs = phase_1_lhs.into_iter().fold(
            F::one(), |acc, mle_ref| {
                acc * mle_ref.bookkeeping_table()[0]
            }
        );
        let last_rhs = phase_1_rhs.into_iter().fold(
            F::one(), |acc, mle_ref| {
                acc * mle_ref.bookkeeping_table()[0]
            }
        );

        // transition into binding y
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

            // bind y, the right side of the sum
            let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_phase2).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                challenges.push(challenge);
                let eval = prove_round(round + self.num_copy_bits, challenge, phase_2_lhs, phase_2_rhs).unwrap();
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
        let num_v = self.rhs.num_vars();

        // first round check
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

        // final round of sumcheck 
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        let ([_, lhs], _) = self.phase_1_mles.as_mut().unwrap();
        let (_, [_, rhs]) = self.phase_2_mles.as_mut().unwrap();
        rhs.fix_variable(num_v - 1 + self.num_copy_bits, final_chal);
        let bound_lhs = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();
        let bound_rhs = check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap();

        // compute the sum over all the variables of the gate function
        let beta_u = BetaTable::new((first_u_challenges.clone(), bound_lhs)).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), bound_rhs)).unwrap();
        let beta_g = self.beta_g.as_ref().unwrap();
        let f_1_uv = self.nonzero_gates.clone().into_iter().fold(
            F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                acc + gz * ux * vy
            }
        );
        
        // get the fully evaluated "expression"
        let fully_evaluated =  f_1_uv * (bound_lhs + bound_rhs);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // error if this doesn't match the last round of sumcheck
        if fully_evaluated != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }
        
        
        Ok(())
    }

    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        let mut claims: Vec<(LayerId, Claim<F>)> = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];

        // check the left side of the sum (f2(u)) against the challenges made to bind that variable
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

        // check the right side of the sum (f3(v)) against the challenges made to bind that variable
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

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        todo!()
    }

    fn get_wlx_evaluations(&self, claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize) -> Result<Vec<F>, ClaimError> {
        todo!()
    }

    fn get_enum(self) -> crate::layer::layer_enum::LayerEnum<F, Self::Transcript> {
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
    #[error("copy phase init error")]
    CopyPhaseInitError,
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
    beta_g: Option<BetaTable<F>>,
    phase_1_mles: Option<([DenseMleRef<F>; 2], [DenseMleRef<F>; 1])>,
    phase_2_mles: Option<([DenseMleRef<F>; 1], [DenseMleRef<F>; 2])>,
    num_copy_bits: usize,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> AddGate<F, Tr> {

    /// new addgate mle 
    pub fn new(layer_id: LayerId, nonzero_gates: Vec<(usize, usize, usize)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>, num_copy_bits: usize) -> AddGate<F, Tr> {
        AddGate {
            layer_id: layer_id,
            nonzero_gates: nonzero_gates,
            lhs: lhs,
            rhs: rhs,
            beta_g: None,
            phase_1_mles: None,
            phase_2_mles: None,
            num_copy_bits: num_copy_bits,
            _marker: PhantomData,
        }
    }

    fn set_beta_g(&mut self, betag: BetaTable<F>) {
        self.beta_g = Some(betag);
    }

    /// bookkeeping tables necessary for binding x
    fn set_phase_1(&mut self, (mle_l, mle_r): ([DenseMleRef<F>; 2], [DenseMleRef<F>; 1])) {
        self.phase_1_mles = Some((mle_l, mle_r));
    }

    /// bookkeeping tables necessary for binding y
    fn set_phase_2(&mut self, (mle_l, mle_r): ([DenseMleRef<F>; 1], [DenseMleRef<F>; 2])) {
        self.phase_2_mles = Some((mle_l, mle_r));
    }

    /// initialize bookkeeping tables for phase 1 of sumcheck
    pub fn init_phase_1(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {

        let beta_g = BetaTable::new(claim).unwrap();
        self.set_beta_g(beta_g.clone());

        // we start indexing at the number of copy bits, because once you get to non-batched add gate, those should be bound
        self.lhs.index_mle_indices(self.num_copy_bits);
        let num_x = self.lhs.num_vars();

        // bookkeeping tables according to libra, set everything to zero for now (we are summing over y so size is 2^(num_x))
        let mut a_hg_lhs = vec![F::zero(); 1 << num_x];
        let mut a_hg_rhs = vec![F::zero(); 1 << num_x];

        // use the gate function to populate the table using beta table 
        self.nonzero_gates.clone().into_iter().for_each(
            |(z_ind, x_ind, y_ind)| {
                let adder = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                a_hg_lhs[x_ind] = a_hg_lhs[x_ind] + adder;
                a_hg_rhs[x_ind] = a_hg_rhs[x_ind] + (adder * *self.rhs.bookkeeping_table().get(y_ind).unwrap_or(&F::zero()));
            }
        );

        let mut phase_1_lhs = [DenseMle::new_from_raw(a_hg_lhs, LayerId::Input, None).mle_ref(), self.lhs.clone()];
        let mut phase_1_rhs = [DenseMle::new_from_raw(a_hg_rhs, LayerId::Input, None).mle_ref()];
        index_mle_indices_gate(phase_1_lhs.as_mut(), self.num_copy_bits);
        index_mle_indices_gate(phase_1_rhs.as_mut(), self.num_copy_bits);
        self.set_phase_1((phase_1_lhs.clone(), phase_1_rhs.clone()));

        // returns the first sumcheck message
        compute_sumcheck_message_gate(&phase_1_lhs, &phase_1_rhs, self.num_copy_bits)
    }

    /// initialize bookkeeping tables for phase 2 of sumcheck
    pub fn init_phase_2(&mut self, u_claim: Claim<F>, f_at_u: F) -> Result<Vec<F>, GateError> {

        let beta_g = self.beta_g.as_ref().expect("beta table should be initialized by now");
        
        // create a beta table according to the challenges used to bind the x variables
        let beta_u = BetaTable::new(u_claim).unwrap();
        let num_y = self.rhs.num_vars();

        // bookkeeping table where we now bind y, so size is 2^(num_y)
        let mut a_f1_lhs = vec![F::zero(); 1 << num_y];
        let mut a_f1_rhs = vec![F::zero(); 1 << num_y];
        // populate the bookkeeping table
        self.nonzero_gates.clone().into_iter().for_each(
            |(z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let adder = gz * ux;
                a_f1_lhs[y_ind] = a_f1_lhs[y_ind] + (adder * f_at_u);
                a_f1_rhs[y_ind] = a_f1_rhs[y_ind] + adder;
            }
        );
        let mut phase_2_lhs = [DenseMle::new_from_raw(a_f1_lhs, LayerId::Input, None).mle_ref()];
        let mut phase_2_rhs = [DenseMle::new_from_raw(a_f1_rhs, LayerId::Input, None).mle_ref(), self.rhs.clone()];
        index_mle_indices_gate(phase_2_lhs.as_mut(), self.num_copy_bits);
        index_mle_indices_gate(phase_2_rhs.as_mut(), self.num_copy_bits);
        self.set_phase_2((phase_2_lhs.clone(), phase_2_rhs.clone()));

        // return the first sumcheck message of this phase
        compute_sumcheck_message_gate(&phase_2_lhs, &phase_2_rhs, self.num_copy_bits)
    }

    /// dummy sumcheck prover for this, testing purposes
    fn dummy_prove_rounds(&mut self, claim: Claim<F>, rng: &mut impl Rng) -> Result<Vec<(Vec<F>, Option<F>)>, GateError> {

        // initialization
        let first_message = self.init_phase_1(claim).expect("could not evaluate original lhs and rhs");
        let (phase_1_lhs, phase_1_rhs) = self.phase_1_mles.as_mut().ok_or(GateError::Phase1InitError).unwrap();

        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        // number of variables in f2 (or the left sum) is how many rounds are in phase 1
        let num_rounds_phase1 = self.lhs.num_vars();
        
        // sumcheck rounds (binding x)
        for round in 1..(num_rounds_phase1) {
            challenge = Some(F::rand(rng));
            //challenge = Some(F::one());
            let chal = challenge.unwrap();
            challenges.push(chal);
            let eval = prove_round(round + self.num_copy_bits, chal, phase_1_lhs, phase_1_rhs).unwrap();
            messages.push((eval, challenge));
        }

        // do the final binding of phase 1
        let final_chal = F::rand(rng);
        //let final_chal = F::from(2_u64);
        challenges.push(final_chal);
        fix_var_gate(phase_1_lhs, num_rounds_phase1 - 1 + self.num_copy_bits, final_chal);
        fix_var_gate(phase_1_rhs, num_rounds_phase1 - 1 + self.num_copy_bits, final_chal);
        let f_2 = &phase_1_lhs[1];

        if f_2.bookkeeping_table.len() == 1 {
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = (challenges.clone(), F::zero());
            
            // first message of the next phase includes the random challenge from the last phase 
            // (this transition checks that we did the bookkeeping optimization correctly between each phase)
            let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();
            let (phase_2_lhs, phase_2_rhs) = self.phase_2_mles.as_mut().ok_or(GateError::Phase2InitError).unwrap();
            messages.push((first_message, Some(final_chal)));

            
            // number of rounds in phase 2 is number of variables in the right sum, binding y
            let num_rounds_phase2 = self.rhs.num_vars();

            // sumcheck rounds (binding y)
            for round in 1..num_rounds_phase2 {
                challenge = Some(F::rand(rng));
                //challenge = Some(F::one());
                let chal = challenge.unwrap();
                challenges.push(chal);
                let eval = prove_round(round + self.num_copy_bits, chal, phase_2_lhs, phase_2_rhs).unwrap();
                messages.push((eval, challenge));
            }
            
            Ok(messages)
        }
        else {
            Err(GateError::Phase1InitError)
        }
        
    }

    /// dummy verifier for dummy sumcheck, testing purposes
    fn dummy_verify_rounds(
        &mut self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
        claim: Claim<F>,
    ) -> Result<(), VerifyError> {

        // first message evals
        let mut prev_evals = &messages[0].0;
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let num_u = self.lhs.num_vars();
        let num_v = self.rhs.num_vars();

        // total number of rounds -- verifier sees no difference between phase 1 and phase 2
        let num_rounds = num_u + num_v;

        // first round checked here
        let claimed_val = messages[0].0[0] + messages[0].0[1];
        if claimed_val != claim.1 {
            dbg!("first check failed");
            dbg!(claimed_val);
            dbg!(-claimed_val);
            return Err(VerifyError::SumcheckBad);
        }
    
        // --- Go through sumcheck messages + (FS-generated) challenges ---
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

        let final_chal = F::rand(rng);
        //let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        // also bind the final challenge (this is in phase 2) to f3 (the right sum)
        let ([_, lhs], _) = self.phase_1_mles.as_mut().unwrap();
        let (_, [_, rhs]) = self.phase_2_mles.as_mut().unwrap();
        rhs.fix_variable(num_v - 1 + self.num_copy_bits, final_chal);
        let f_2_u = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();
        let f_3_v = check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap();

        // evaluate the gate function at the bound points!!!
        let beta_u = BetaTable::new((first_u_challenges.clone(), f_2_u)).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), f_3_v)).unwrap();
        let beta_g = self.beta_g.as_ref().unwrap();
        let f_1_uv = self.nonzero_gates.clone().into_iter().fold(
            F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                acc + gz * ux * vy
            }
        );

        // equation for the expression
        let last_v_bound = f_1_uv * (f_2_u + f_3_v);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final round of sumcheck check
        if last_v_bound != prev_at_r {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }

}

/// evaluate_mle_ref_product without beta tables........
fn evaluate_mle_ref_product_gate<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    independent_variable: bool,
    degree: usize,
) -> Result<Evals<F>, MleError> {
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

        Ok(Evals(evals))
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

        Ok(Evals(vec![sum; degree]))
    }
}

/// compute the beta value after everything has been bound
fn fully_bind_beta<F: FieldExt>(
    claim_chals: Vec<F>,
    challenges: Vec<F>,
) -> F {
    challenges
            .into_iter()
            .enumerate()
            .fold(
                F::one(),
                |acc, (round_idx, challenge)| {
                acc * ((claim_chals[round_idx] * challenge) + ((F::one() - claim_chals[round_idx]) * (F::one() - challenge)))
            })
}

/// checks whether mle was bound correctly to all the challenge points!!!!!!!!!!
fn check_fully_bound<F: FieldExt>(
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

        let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

        if indices.as_slice() != &challenges[..] {
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

/// index mle indices for an array of mles
fn index_mle_indices_gate<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    index: usize,
) {
    mle_refs.iter_mut().for_each(
        |mle_ref| {
            mle_ref.index_mle_indices(index);
        }
    )
}

/// fix variable for an array of mles
fn fix_var_gate<F: FieldExt>(
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

/// compute sumcheck message without a beta table!!!!!!!!!!!!!!
fn compute_sumcheck_message_gate<F: FieldExt>(
    lhs: &[impl MleRef<F = F>],
    rhs: &[impl MleRef<F = F>],
    round_index: usize,
) -> Result<Vec<F>, GateError> {

    // for gate mles, degree always 2 for left and right side because on one side we are taking the product of two bookkkeping tables
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
    .ok_or(GateError::EmptyMleList)?;    
    let eval_rhs = evaluate_mle_ref_product_gate(rhs, independent_variable_rhs, degree).unwrap();
    let eval = eval_lhs + eval_rhs;

    let Evals(evaluations) = eval;

    Ok(evaluations)
}

///Computes a round of the sumcheck protocol on this Layer
fn prove_round<F: FieldExt>(round_index: usize, challenge: F, lhs: &mut [impl MleRef<F = F>], rhs: &mut [impl MleRef<F = F>]) -> Result<Vec<F>, GateError> {
    fix_var_gate(lhs, round_index - 1, challenge);
    fix_var_gate(rhs, round_index - 1, challenge);
    compute_sumcheck_message_gate(lhs, rhs, round_index)
}



impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for AddGateBatched<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(&mut self, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<SumcheckProof<F>, LayerError> {
        // initialization, first message comes from here
        let first_message = self.init_copy_phase(claim.clone()).expect("could not evaluate original lhs and rhs");
        let (beta_g, [a_f2, a_f3]) = self.copy_phase_mles.as_mut().ok_or(GateError::CopyPhaseInitError).unwrap();
        let (mut lhs, mut rhs) = (&mut self.lhs, &mut self.rhs);
        
        let mut challenges: Vec<F> = vec![];

        // new bits is the number of bits representing which copy of the gate we are looking at
        transcript
        .append_field_elements("Initial Sumcheck evaluations", &first_message)
        .unwrap();
        let num_rounds_copy_phase = self.new_bits;

        // do the first copy bits number sumcheck rounds
        let mut sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
        .chain((1..num_rounds_copy_phase).map(|round| {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
            challenges.push(challenge);
            let eval = prove_round_copy(a_f2, a_f3, &mut lhs, &mut rhs, beta_g, round, challenge).unwrap();
            transcript
                .append_field_elements("Sumcheck evaluations", &eval)
                .unwrap();
            Ok::<_, LayerError>(eval)
        }))
        .try_collect()?;

        // bind the final challenge, update the final beta table
        let final_chal_copy = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal_copy);

        // fix the variable and everything as you would in the last round of sumcheck
        // the evaluations from this is what you return from the first round of sumcheck in the next phase!
        a_f2.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        a_f3.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        beta_g.beta_update(num_rounds_copy_phase - 1, final_chal_copy).unwrap();
        self.lhs.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g.table.bookkeeping_table.len() == 1 {
            let beta_g2 = beta_g.table.bookkeeping_table()[0];
            let next_claims = (self.g1_challenges.clone().unwrap(), F::zero());

            // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
            // this essentially takes in the two mles bound only at the copy bits
            let mut reduced_gate: AddGate<F, Tr> = AddGate::new(self.layer_id.clone(), self.nonzero_gates.clone(), self.lhs.clone(), self.rhs.clone(), self.new_bits);
            let next_messages = reduced_gate.prove_rounds(next_claims, transcript).unwrap();
        
            // we scale the messages by the bound beta table (g2, w) where g2 is the challenge
            // from the claim on the copy bits and w is the challenge point we bind the copy bits to
            let scaled_next_messages: Vec<Vec<F>> = next_messages.0.into_iter().map(
                |evals| {
                    evals.into_iter().map(
                        |eval| {
                            eval * beta_g2
                        }
                    ).collect_vec()
                }
            ).collect();
            sumcheck_rounds.extend(scaled_next_messages);
            return Ok(sumcheck_rounds.into());
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
        let mut first_copy_challenges = vec![];
        let num_u = self.num_vars_l.unwrap();
        let num_v = self.num_vars_r.unwrap();

        // first check!!!!
        let claimed_val = sumcheck_rounds[0][0] + sumcheck_rounds[0][1];
        if claimed_val != claim.1 {
            return Err(LayerError::VerificationError(VerificationError::SumcheckStartFailed));
        }

        // check each of the messages
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

            // we want to separate the challenges into which ones are from the copy bits, which ones
            // are for binding x, and which are for binding y (non-batched) 
            if (..(self.new_bits + 1)).contains(&i) {
                first_copy_challenges.push(challenge);
            }
            else if (..(num_u + 1)).contains(&i) {
                first_u_challenges.push(challenge);
            }
            else {
                last_v_challenges.push(challenge);
            }
        }

        // final round of sumcheck
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        // this belongs in the last challenge bound to y
        last_v_challenges.push(final_chal);

        // we want to grab the mutated bookkeeping tables from the "reduced_gate", this is the non-batched version
        let ([_, lhs_reduced], _) = self.reduced_gate.as_ref().unwrap().phase_1_mles.as_ref().unwrap().clone();
        fix_var_gate(&mut self.reduced_gate.as_mut().unwrap().phase_2_mles.as_mut().unwrap().1, num_v - 1, final_chal);
        let (_, [_, rhs_reduced]) = self.reduced_gate.as_ref().unwrap().phase_2_mles.as_ref().unwrap().clone();

        // since the original mles are batched, the challenges are the concat of the copy bits and the variable bound bits
        let lhs_challenges = [first_copy_challenges.clone().as_slice(), first_u_challenges.clone().as_slice()].concat();
        let rhs_challenges = [first_copy_challenges.clone().as_slice(), last_v_challenges.clone().as_slice()].concat();

        // compute the gate function bound at those variables
        let beta_u = BetaTable::new((first_u_challenges.clone(), F::zero())).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), F::zero())).unwrap();
        let beta_g = BetaTable::new((self.g1_challenges.clone().unwrap(), F::zero())).unwrap();
        let f_1_uv = self.nonzero_gates.clone().into_iter().fold(
            F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                acc + gz * ux * vy
            }
        );

        // check that the original mles have been bound correctly -- this is what we get from the reduced gate
        check_fully_bound(&mut [lhs_reduced.clone()], lhs_challenges.clone()).unwrap();
        check_fully_bound(&mut [rhs_reduced.clone()], rhs_challenges.clone()).unwrap();
        let f2_bound = lhs_reduced.bookkeeping_table()[0];
        let f3_bound = rhs_reduced.bookkeeping_table()[0];
        let beta_bound = fully_bind_beta(self.g2_challenges.clone().unwrap(), first_copy_challenges);

        // compute the final result of the bound expression
        let final_result = beta_bound * (f_1_uv * (f2_bound + f3_bound));

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final check in sumcheck
        if final_result != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    ///Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        // we are going to grab the claims from the reduced gate -- this is where the mles are finally mutated
        let ([_, lhs_reduced], _) = self.reduced_gate.as_ref().unwrap().phase_1_mles.as_ref().unwrap().clone();
        let (_, [_, rhs_reduced]) = self.reduced_gate.as_ref().unwrap().phase_2_mles.as_ref().unwrap().clone();

        let mut claims: Vec<(LayerId, (Vec<F>, F))> = vec![];

        // grab the claim on the left sum
        let mut fixed_mle_indices_u: Vec<F> = vec![];
        for index in lhs_reduced.mle_indices() {
            fixed_mle_indices_u.push(index.val().ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?);
        }
        let val = lhs_reduced.bookkeeping_table()[0];
        claims.push((self.id().clone(), (fixed_mle_indices_u, val)));
        
        // grab the claim on the right sum
        let mut fixed_mle_indices_v: Vec<F> = vec![];
        for index in rhs_reduced.mle_indices() {
            fixed_mle_indices_v.push(index.val().ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?);
        }
        let val = rhs_reduced.bookkeeping_table()[0];
        claims.push((self.id().clone(), (fixed_mle_indices_v, val)));
        
        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        todo!()
    }

    fn get_wlx_evaluations(&self, claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize) -> Result<Vec<F>, ClaimError> {
        todo!()
    }

    fn get_enum(self) -> crate::layer::layer_enum::LayerEnum<F, Self::Transcript> {
        todo!()
    }
}



/// batched impl for gate
pub struct AddGateBatched<F: FieldExt, Tr: Transcript<F>> {
    new_bits: usize,
    nonzero_gates: Vec<(usize, usize, usize)>,
    lhs: DenseMleRef<F>,
    rhs: DenseMleRef<F>,
    num_vars_l: Option<usize>,
    num_vars_r: Option<usize>,
    copy_phase_mles: Option<(BetaTable<F>, [DenseMleRef<F>; 2])>,
    g1_challenges: Option<Vec<F>>,
    g2_challenges: Option<Vec<F>>,
    layer_id: LayerId,
    reduced_gate: Option<AddGate<F, Tr>>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> AddGateBatched<F, Tr> {
    /// new batched addgate thingy
    pub fn new(new_bits: usize, nonzero_gates: Vec<(usize, usize, usize)>, lhs: DenseMleRef<F>, rhs: DenseMleRef<F>, layer_id: LayerId) -> Self {
        AddGateBatched {
            new_bits: new_bits,
            nonzero_gates: nonzero_gates,
            lhs: lhs,
            rhs: rhs,
            num_vars_l: None,
            num_vars_r: None,
            copy_phase_mles: None,
            g1_challenges: None,
            g2_challenges: None,
            layer_id: layer_id,
            reduced_gate: None,
            _marker: PhantomData,
        }
    }

    /// sets all the attributes after the "copy phase" is initialized
    fn set_copy_phase(&mut self, mles: (BetaTable<F>, [DenseMleRef<F>; 2]), g1_challenges: Vec<F>, g2_challenges: Vec<F>, lhs_num_vars: usize, rhs_num_vars: usize) {
        self.copy_phase_mles = Some(mles);
        self.g1_challenges = Some(g1_challenges);
        self.g2_challenges = Some(g2_challenges);
        self.num_vars_l = Some(lhs_num_vars);
        self.num_vars_r = Some(rhs_num_vars);
    }

    /// initializes the copy phase
    fn init_copy_phase(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        let (challenges, _) = claim;
        let mut g2_challenges: Vec<F> = vec![];
        let mut g1_challenges: Vec<F> = vec![];

        // we split the claim challenges into two -- the first copy_bits number of challenges are referred
        // to as g2, and the rest are referred to as g1. this distinguishes batching from non-batching internally
        challenges.iter().enumerate().for_each(
            |(bit_idx, challenge)| {
                if bit_idx < self.new_bits {
                    g2_challenges.push(*challenge);
                }
                else {
                    g1_challenges.push(*challenge);
                }
            }
        );

        // create two separate beta tables for each, as they are handled differently
        let mut beta_g2 = BetaTable::new((g2_challenges.clone(), F::zero())).unwrap();
        let beta_g1 = BetaTable::new((g1_challenges.clone(), F::zero())).unwrap();

        // the bookkeeping tables of this phase must have size 2^copy_bits (refer to vibe check B))
        let num_copy_vars = 1 << self.new_bits;
        let mut a_f2 = vec![F::zero(); num_copy_vars];
        let mut a_f3 = vec![F::zero(); num_copy_vars];

        // populate the bookkeeping tables
        (0..num_copy_vars).into_iter().for_each(|idx|
            {
                let mut adder_f2 = F::zero();
                let mut adder_f3 = F::zero();
                // we need to compute the value of the gate function at each of these points before adding them
                self.nonzero_gates.clone().into_iter().for_each(
                        |(z_ind, x_ind, y_ind)| {
                            let gz = *beta_g1.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                            let f2_val = *self.lhs.bookkeeping_table().get(idx + (x_ind * num_copy_vars)).unwrap_or(&F::zero());
                            let f3_val = *self.rhs.bookkeeping_table().get(idx + (y_ind * num_copy_vars)).unwrap_or(&F::zero());
                            adder_f2 += gz * f2_val;
                            adder_f3 += gz * f3_val;
                        }
                    );
                a_f2[idx] += adder_f2; 
                a_f3[idx] += adder_f3; 
            });
        
        let mut a_f2_mle = DenseMle::new_from_raw(a_f2, LayerId::Input, None).mle_ref();
        a_f2_mle.index_mle_indices(0);
        let mut a_f3_mle = DenseMle::new_from_raw(a_f3, LayerId::Input, None).mle_ref();
        a_f3_mle.index_mle_indices(0);
        beta_g2.table.index_mle_indices(0);
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);


        self.set_copy_phase((beta_g2.clone(), [a_f2_mle.clone(), a_f3_mle.clone()]), g1_challenges, g2_challenges, self.lhs.num_vars(), self.rhs.num_vars());
        
        // result of initializing is the first sumcheck message!
        compute_sumcheck_message_copy(&mut beta_g2, &mut a_f2_mle, &mut a_f3_mle,  0)
    }

    /// a prove rounds function for testing purposes
    fn dummy_prove_rounds(&mut self, claim: Claim<F>, rng: &mut impl Rng) -> Vec<(Vec<F>, Option<F>)> {
        // initialization
        let first_message = self.init_copy_phase(claim.clone()).expect("could not evaluate original lhs and rhs");
        let (beta_g, [a_f2, a_f3]) = self.copy_phase_mles.as_mut().ok_or(GateError::CopyPhaseInitError).unwrap();
        let (mut lhs, mut rhs) = (&mut self.lhs, &mut self.rhs);
        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        let num_rounds_copy_phase = self.new_bits;

        // sumcheck rounds -- over here we bind the copy bits
        for round in 1..(num_rounds_copy_phase) {
            challenge = Some(F::rand(rng));
            //let challenge = Some(F::one());
            let chal = challenge.unwrap();
            challenges.push(chal);
            
            let evals = prove_round_copy(a_f2, a_f3, &mut lhs, &mut rhs, beta_g, round, chal).unwrap();
            messages.push((evals, challenge));
        }

        // final challenge of binding the copy bits
        //let final_chal = F::one();
        let final_chal = F::rand(rng);
        challenges.push(final_chal);
        a_f2.fix_variable(num_rounds_copy_phase - 1, final_chal);
        a_f3.fix_variable(num_rounds_copy_phase - 1, final_chal);
        beta_g.beta_update(num_rounds_copy_phase - 1, final_chal).unwrap();
        self.lhs.fix_variable(num_rounds_copy_phase - 1, final_chal);
        self.rhs.fix_variable(num_rounds_copy_phase - 1, final_chal);

        // grab the bound beta value
        let beta_g2 = beta_g.table.bookkeeping_table()[0];
        let next_claims = (self.g1_challenges.clone().unwrap(), F::zero());

        // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
        let reduced_gate: AddGate<F, Tr> = AddGate::new(self.layer_id.clone(), self.nonzero_gates.clone(), self.lhs.clone(), self.rhs.clone(), self.new_bits);
        self.reduced_gate = Some(reduced_gate);
        let mut next_messages = self.reduced_gate.as_mut().unwrap().dummy_prove_rounds(next_claims, rng).unwrap();
        let (next_first, _) = &mut next_messages[0];

        // mutate the message from the non-batched case to include the last challenge from copy phase
        next_messages[0] = (next_first.clone(), Some(final_chal));

        // scale it by the bound beta value
        let scaled_next_messages: Vec<(Vec<F>, Option<F>)> = next_messages.into_iter().map(
            |(evals, chal)| {
                let scaled_evals = evals.into_iter().map(
                    |eval| {
                        eval * beta_g2
                    }
                ).collect_vec();
                (scaled_evals, chal)
            }
        ).collect();

        messages.extend(scaled_next_messages.into_iter());

        messages
    }

    fn dummy_verify_rounds(
        &mut self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
        claim: Claim<F>,
    ) -> Result<(), VerifyError> {
        let mut prev_evals = &messages[0].0;
    
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let mut first_copy_challenges = vec![];
        let num_u = self.num_vars_l.unwrap();
        let num_v = self.num_vars_r.unwrap();
        let num_rounds = num_u + num_v - self.new_bits;

        // first check!!
        let claimed_val = messages[0].0[0] + messages[0].0[1];
        if claimed_val != claim.1 {
            dbg!("first check failed");
            dbg!(claimed_val);
            return Err(VerifyError::SumcheckBad);
        }
    
        // --- Go through sumcheck messages + (FS-generated) challenges ---
        for i in 1..(num_rounds) {
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            let chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                dbg!("fail at round ", &i);
                dbg!(prev_at_r);
                dbg!(&curr_evals);
                dbg!(curr_evals[0] + curr_evals[1]);
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);

            if (..(self.new_bits + 1)).contains(&i) {
                first_copy_challenges.push(chal);
            }
            else if (..(num_u + 1)).contains(&i) {
                first_u_challenges.push(chal);
            }
            else {
                last_v_challenges.push(chal);
            }
        }

        // last challenge
        let final_chal = F::rand(rng);
        //let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);
        // fix the y variable in the reduced gate at this point too
        fix_var_gate(&mut self.reduced_gate.as_mut().unwrap().phase_2_mles.as_mut().unwrap().1, num_v - 1, final_chal);

        // compute the gate function evaluated at bound variables
        let beta_u = BetaTable::new((first_u_challenges.clone(), F::zero())).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), F::zero())).unwrap();
        let beta_g = BetaTable::new((self.g1_challenges.clone().unwrap(), F::zero())).unwrap();
        let f_1_uv = self.nonzero_gates.clone().into_iter().fold(
            F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                acc + gz * ux * vy
            }
        );


        // honestly just checking if get_claims() computes correctly, use this to get the bound f_2 and f_3 values
        let claims = self.get_claims().unwrap().clone();
        let [(_, (_, f2_bound)), (_, (_, f3_bound))] = claims.as_slice() else { return Err(VerifyError::SumcheckBad) };

        let beta_bound = fully_bind_beta(self.g2_challenges.clone().unwrap(), first_copy_challenges);

        // evaluate the gate expression 
        let final_result = beta_bound * (f_1_uv * (*f2_bound + *f3_bound));

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final check of sumcheck
        if final_result != prev_at_r {
            dbg!("last round fail");
            dbg!(prev_at_r);
            dbg!(final_result);
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }

}

/// computes the sumcheck message for batched gate mle
fn compute_sumcheck_message_copy<F: FieldExt>(beta: &mut BetaTable<F>, lhs: &mut DenseMleRef<F>, rhs: &mut DenseMleRef<F>, round_index: usize,) -> Result<Vec<F>, GateError> {
    
    // degree is 2 because we use a beta table
    let degree = 2;
    let independent_lhs = lhs.mle_indices().contains(&MleIndex::IndexedBit(round_index));
    let independent_rhs = rhs.mle_indices().contains(&MleIndex::IndexedBit(round_index));
    
    let evals_lhs = evaluate_mle_ref_product(&[lhs.clone()], independent_lhs, degree, beta.clone().table).unwrap();
    let evals_rhs = evaluate_mle_ref_product(&[rhs.clone()], independent_rhs, degree, beta.clone().table).unwrap();

    let eval = evals_lhs + evals_rhs;
    let Evals(evaluations) = eval;
    
    Ok(evaluations)
}

/// does all the necessary updates when proving a round for batched gate mles
fn prove_round_copy<F: FieldExt>(phase_lhs: &mut DenseMleRef<F>, phase_rhs: &mut DenseMleRef<F>, lhs: &mut DenseMleRef<F>, rhs: &mut DenseMleRef<F>, beta: &mut BetaTable<F>, round_index: usize, challenge: F) -> Result<Vec<F>, GateError> {
    phase_lhs.fix_variable(round_index - 1, challenge);
    phase_rhs.fix_variable(round_index - 1, challenge);
    beta.beta_update(round_index - 1, challenge).unwrap();
    // need to separately update these because the phase_lhs and phase_rhs has no version of them
    lhs.fix_variable(round_index - 1, challenge);
    rhs.fix_variable(round_index - 1, challenge);
    compute_sumcheck_message_copy(beta, phase_lhs, phase_rhs, round_index)
}

#[cfg(test)]
mod test {
    use crate::mle::dense::DenseMle;
    use crate::transcript::poseidon_transcript::PoseidonTranscript;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    /// non-batched test
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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

        let rhs_v = vec![
            Fr::from(51395810),
            Fr::from(2),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0), nonzero_gates, lhs_mle_ref, rhs_mle_ref, 0);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    /// non-batched test
    fn test_sumcheck_2() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(0),Fr::from(1), Fr::from(0)], Fr::from(2));
        let nonzero_gates = vec![
            (1, 0, 1),
            (3, 0, 2),
            (2, 3, 4),
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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0),  nonzero_gates, lhs_mle_ref, rhs_mle_ref, 0);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());


    }

    #[test]
    /// non-batched test
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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut gate_mle: AddGate<Fr, PoseidonTranscript<Fr>> = AddGate::new(LayerId::Layer(0), nonzero_gates, lhs_mle_ref, rhs_mle_ref, 0);
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_1() {

        let mut rng = test_rng();

        let claim = (vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(0), Fr::from(0)], Fr::from(0));
        let new_bits = 1;
        let nonzero_gates = vec![
            (1, 1, 1),
            ];

        let lhs_v = vec![
            Fr::from(0),
            Fr::from(5),
            Fr::from(1),
            Fr::from(2),
        ];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

        let rhs_v = vec![
            Fr::from(0),
            Fr::from(5),
            Fr::from(51395810),
            Fr::from(2),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut batched_gate_mle: AddGateBatched<Fr, PoseidonTranscript<Fr>> = AddGateBatched::new(new_bits, nonzero_gates, lhs_mle_ref, rhs_mle_ref, LayerId::Layer(0));
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_2() {

        let mut rng = test_rng();
        let new_bits = 2;

        let claim = (vec![Fr::from(1), Fr::from(1),Fr::from(1),], Fr::from(2));
        let nonzero_gates = vec![
            (1, 1, 1),
            (0, 0, 1),
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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut batched_gate_mle: AddGateBatched<Fr, PoseidonTranscript<Fr>> = AddGateBatched::new(new_bits, nonzero_gates, lhs_mle_ref, rhs_mle_ref, LayerId::Layer(0));
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_3() {

        let mut rng = test_rng();
        let new_bits = 1;

        let claim = (vec![Fr::from(3), Fr::from(1),Fr::from(1),], Fr::from(22));
        let nonzero_gates = vec![
            (3, 1, 1),
            (2, 1, 0),
            ];

        let lhs_v = vec![
            Fr::from(0), 
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
            Fr::from(3),
        ];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut batched_gate_mle: AddGateBatched<Fr, PoseidonTranscript<Fr>> = AddGateBatched::new(new_bits, nonzero_gates, lhs_mle_ref, rhs_mle_ref, LayerId::Layer(0));
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_4() {

        let mut rng = test_rng();
        let new_bits = 2;

        let claim = (vec![Fr::from(1), Fr::from(1),Fr::from(1),Fr::from(2)], Fr::from(4));
        let nonzero_gates = vec![
            (3, 1, 1),
            (2, 1, 0),
            (1, 0, 1),
            ];

        let lhs_v = vec![
            Fr::from(0), 
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
            Fr::from(3),
            Fr::from(0), 
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(3),
            Fr::from(5),
            Fr::from(7),
            Fr::from(3),
        ];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input, None).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(4),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input, None).mle_ref();

        let mut batched_gate_mle: AddGateBatched<Fr, PoseidonTranscript<Fr>> = AddGateBatched::new(new_bits, nonzero_gates, lhs_mle_ref, rhs_mle_ref, LayerId::Layer(0));
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());

    }

}
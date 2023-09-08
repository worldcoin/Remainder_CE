use std::marker::PhantomData;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rand::Rng;
use remainder_shared_types::{FieldExt, transcript::Transcript};
use serde::{Serialize, Deserialize};

use crate::{layer::{Layer, Claim, LayerError, VerificationError, LayerId, LayerBuilder, claims::ClaimError, layer_enum::LayerEnum}, prover::SumcheckProof, sumcheck::{evaluate_at_a_point, VerifyError, Evals}, mle::{beta::{BetaTable, compute_beta_over_two_challenges}, mulgate::check_fully_bound, dense::DenseMle}};

use super::{mulgate::{GateError, compute_full_mulgate, MulGate, fix_var_gate}, dense::DenseMleRef, MleRef, MleIndex};
use crate::sumcheck::evaluate_mle_ref_product;
use rayon::{iter::IntoParallelIterator, prelude::ParallelIterator};

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for MulGateBatched<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        // initialization, first message comes from here
        let first_message = self
            .init_copy_phase(claim.clone())
            .expect("could not evaluate original lhs and rhs");
        let (beta_g, [a_f2]) = self
            .copy_phase_mles
            .as_mut()
            .ok_or(GateError::CopyPhaseInitError)
            .unwrap();
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
                let eval =
                    prove_round_copy(&mut [a_f2.clone()], &mut lhs, &mut rhs, beta_g, round, challenge)
                        .unwrap();
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
        // a_f2.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        // a_f3.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        beta_g
            .beta_update(num_rounds_copy_phase - 1, final_chal_copy)
            .unwrap();
        self.lhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g.table.bookkeeping_table.len() == 1 {
            let beta_g2 = beta_g.table.bookkeeping_table()[0];
            let next_claims = (self.g1_challenges.clone().unwrap(), F::zero());

            // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
            // this essentially takes in the two mles bound only at the copy bits
            let mut reduced_gate: MulGate<F, Tr> = MulGate::new(self.layer_id.clone(), self.nonzero_gates.clone(), self.lhs.clone(), self.rhs.clone(), self.new_bits, Some(beta_g2));
            dbg!(&self.lhs.clone());
            dbg!(&self.rhs.clone());
            self.reduced_gate = Some(reduced_gate);
            let next_messages = self.reduced_gate.as_mut().unwrap().prove_rounds(next_claims, transcript).unwrap();

            // we scale the messages by the bound beta table (g2, w) where g2 is the challenge
            // from the claim on the copy bits and w is the challenge point we bind the copy bits to
            // let scaled_next_messages: Vec<Vec<F>> = next_messages
            //     .0
            //     .into_iter()
            //     .map(|evals| evals.into_iter().map(|eval| eval * beta_g2).collect_vec())
            //     .collect();
            sumcheck_rounds.extend(next_messages.0);

            // dbg!(&sumcheck_rounds);

            return Ok(sumcheck_rounds.into());
        } else {
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
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();

        // check each of the messages
        for (i, curr_evals) in sumcheck_rounds.iter().enumerate().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge)
                .map_err(|err| LayerError::InterpError(err))?;
            
            dbg!(&i);
            dbg!(&prev_at_r);
            // dbg!(&prev_evals);
            dbg!(&curr_evals);

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                // return Err(LayerError::VerificationError(
                //     VerificationError::SumcheckFailed,
                // ));
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
            } else if (..(num_u + 1)).contains(&i) {
                first_u_challenges.push(challenge);
            } else {
                last_v_challenges.push(challenge);
            }
        }

        // final round of sumcheck
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        dbg!(final_chal);

        // this belongs in the last challenge bound to y
        last_v_challenges.push(final_chal);

        // we want to grab the mutated bookkeeping tables from the "reduced_gate", this is the non-batched version
        let [_, lhs_reduced] = self.reduced_gate.as_ref().unwrap().phase_1_mles.as_ref().unwrap().clone();
        let [_, rhs_reduced] = self.reduced_gate.as_ref().unwrap().phase_2_mles.as_ref().unwrap().clone();

        // since the original mles are batched, the challenges are the concat of the copy bits and the variable bound bits
        let lhs_challenges = [first_copy_challenges.clone().as_slice(), first_u_challenges.clone().as_slice()].concat();
        let rhs_challenges = [first_copy_challenges.clone().as_slice(), last_v_challenges.clone().as_slice()].concat();

        let g2_challenges = claim.0[..self.new_bits].to_vec();
        let g1_challenges = claim.0[self.new_bits..].to_vec();

        // compute the gate function bound at those variables
        let beta_u = BetaTable::new((first_u_challenges.clone(), F::zero())).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), F::zero())).unwrap();
        let beta_g = BetaTable::new((g1_challenges, F::zero())).unwrap();
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
        let beta_bound = compute_beta_over_two_challenges(&g2_challenges, &first_copy_challenges);

        // compute the final result of the bound expression
        let final_result = beta_bound * (f_1_uv * (f2_bound * f3_bound));

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
        let [_, lhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_1_mles
            .as_ref()
            .unwrap()
            .clone();
        let [_, rhs_reduced] = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_2_mles
            .as_ref()
            .unwrap()
            .clone();

        let mut claims: Vec<(LayerId, (Vec<F>, F))> = vec![];

        // grab the claim on the left sum
        let mut fixed_mle_indices_u: Vec<F> = vec![];
        for index in lhs_reduced.mle_indices() {
            fixed_mle_indices_u.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = lhs_reduced.bookkeeping_table()[0];
        claims.push((self.lhs.get_layer_id(), (fixed_mle_indices_u, val)));

        // grab the claim on the right sum
        let mut fixed_mle_indices_v: Vec<F> = vec![];
        for index in rhs_reduced.mle_indices() {
            fixed_mle_indices_v.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = rhs_reduced.bookkeeping_table()[0];
        claims.push((self.rhs.get_layer_id(), (fixed_mle_indices_v, val)));

        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        unimplemented!()
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: Vec<Vec<F>>,
        claimed_vals: &mut Vec<F>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let num_vars = std::cmp::max(self.lhs.num_vars(), self.rhs.num_vars());
        let num_evals = (num_vars) * (num_claims); //* degree;

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = (num_claims..num_evals)
            .into_iter()
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

                let eval = compute_full_mulgate(
                    new_chal,
                    &mut self.lhs.clone(),
                    &mut self.rhs.clone(),
                    &self.nonzero_gates,
                    self.new_bits,
                );
                eval
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals.clone();
        Ok(wlx_evals)
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        LayerEnum::MulGateBatched(self)
    }
}


/// batched impl for gate
///
/// ## Fields
/// * `new_bits` - number of bits in p2
/// * `nonzero_gates` - Same as `AddGate` (non-batched)
/// * `lhs` - MLEs on the left side to be added.
/// * `rhs` - MLEs on the right side to be added.
/// * `num_vars_l` - Length of `x` (as in f_2(x))
/// * `num_vars_r` - Length of `y` (as in f_3(y))
/// * `copy_phase_mles` - List of MLEs for when we're binding the vars representing the batched bits
/// * `g1_challenges` - Literally g_1
/// * `g2_challenges` - Literally g_2
/// * `layer_id` - GKR layer number
/// * `reduced_gate` - the non-batched gate that this reduces to after the copy phase
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MulGateBatched<F: FieldExt, Tr: Transcript<F>> {
    pub new_bits: usize,
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    pub lhs: DenseMleRef<F>,
    pub rhs: DenseMleRef<F>,
    pub num_vars_l: Option<usize>,
    pub num_vars_r: Option<usize>,
    copy_phase_mles: Option<(BetaTable<F>, [DenseMleRef<F>; 1])>,
    pub g1_challenges: Option<Vec<F>>,
    pub g2_challenges: Option<Vec<F>>,
    pub layer_id: LayerId,
    pub reduced_gate: Option<MulGate<F, Tr>>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> MulGateBatched<F, Tr> {
    /// new batched addgate thingy
    pub fn new(
        new_bits: usize,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        layer_id: LayerId,
    ) -> Self {
        MulGateBatched {
            new_bits,
            nonzero_gates,
            lhs,
            rhs,
            num_vars_l: None,
            num_vars_r: None,
            copy_phase_mles: None,
            g1_challenges: None,
            g2_challenges: None,
            layer_id,
            reduced_gate: None,
            _marker: PhantomData,
        }
    }

    /// sets all the attributes after the "copy phase" is initialized
    fn set_copy_phase(
        &mut self,
        mles: (BetaTable<F>, [DenseMleRef<F>; 1]),
        g1_challenges: Vec<F>,
        g2_challenges: Vec<F>,
        lhs_num_vars: usize,
        rhs_num_vars: usize,
    ) {
        self.copy_phase_mles = Some(mles);
        self.g1_challenges = Some(g1_challenges);
        self.g2_challenges = Some(g2_challenges);
        self.num_vars_l = Some(lhs_num_vars);
        self.num_vars_r = Some(rhs_num_vars);
    }

    /// initializes the copy phase
    ///
    /// ---
    ///
    /// The expression for this phase is as follows (note that we are binding `g_2`, i.e. the batch bits):
    /// * V_i(g_2, g_1) = \sum_{p_2, x, y} \beta(g_2, p_2) f_1(g_1, x, y) (f_2(p_2, x) + f_3(p_2, y))
    ///
    /// We thus need the following bookkeeping tables:
    /// * \beta(g_2, p_2)
    /// * a_f2(p_2) = \sum_{x, y} f_1(g_2, x, y) * f_2(p_2, x) = \sum_{p_2, z, x, y \in N_x} \beta(g_2, z) f_2(p_2, x)
    /// * a_f3(p_2) = \sum_{x, y} f_1(g_2, x, y) * f_3(p_2, y) = \sum_{p_2, z, x, y \in N_x} \beta(g_2, z) f_3(p_2, y)
    ///
    /// Note that --
    /// * The first one is computed via initializing a beta table.
    /// * The second/third ones are computed via iterating over all (sparse) (p_2, z, x, y) points and summing the terms above.
    fn init_copy_phase(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        let (challenges, _) = claim;
        let mut g2_challenges: Vec<F> = vec![];
        let mut g1_challenges: Vec<F> = vec![];

        // we split the claim challenges into two -- the first copy_bits number of challenges are referred
        // to as g2, and the rest are referred to as g1. this distinguishes batching from non-batching internally
        challenges
            .iter()
            .enumerate()
            .for_each(|(bit_idx, challenge)| {
                if bit_idx < self.new_bits {
                    g2_challenges.push(*challenge);
                } else {
                    g1_challenges.push(*challenge);
                }
            });

        // create two separate beta tables for each, as they are handled differently
        let mut beta_g2 = BetaTable::new((g2_challenges.clone(), F::zero())).unwrap();
        let beta_g1 = BetaTable::new((g1_challenges.clone(), F::zero())).unwrap();

        // the bookkeeping tables of this phase must have size 2^copy_bits (refer to vibe check B))
        let num_copy_vars = 1 << self.new_bits;
        let mut a_f2 = vec![F::zero(); num_copy_vars];
        // let mut a_f3 = vec![F::zero(); num_copy_vars];

        // populate the bookkeeping tables
        // TODO!(ryancao): Good optimization here is to parallelize -- I don't think there are any race conditions
        (0..num_copy_vars).into_iter().for_each(|idx| {
            let mut adder_f2 = F::zero();
            // we need to compute the value of the gate function at each of these points before adding them
            self.nonzero_gates
                .clone()
                .into_iter()
                .for_each(|(z_ind, x_ind, y_ind)| {
                    let gz = *beta_g1
                        .table
                        .bookkeeping_table()
                        .get(z_ind)
                        .unwrap_or(&F::zero());
                    let f2_val = *self
                        .lhs
                        .bookkeeping_table()
                        .get(idx + (x_ind * num_copy_vars))
                        .unwrap_or(&F::zero());
                    let f3_val = *self
                        .rhs
                        .bookkeeping_table()
                        .get(idx + (y_ind * num_copy_vars))
                        .unwrap_or(&F::zero());

                    dbg!(&gz);
                    dbg!(&f2_val);
                    dbg!(&f3_val);
                    adder_f2 += gz * f2_val * f3_val
                });
            a_f2[idx] += adder_f2;
            // a_f3[idx] += adder_f3;
        });

        // --- Wrappers over the bookkeeping tables ---
        let mut a_f2_mle = DenseMle::new_from_raw(a_f2, LayerId::Input(0), None).mle_ref();
        a_f2_mle.index_mle_indices(0);
        // let mut a_f3_mle = DenseMle::new_from_raw(a_f3, LayerId::Input(0), None).mle_ref();
        // a_f3_mle.index_mle_indices(0);
        beta_g2.table.index_mle_indices(0);

        // index original bookkeeping tables to send over to the non-batched mul gate after the copy phase
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // --- Sets self internal state ---
        self.set_copy_phase(
            (beta_g2.clone(), [a_f2_mle.clone()]),
            g1_challenges,
            g2_challenges,
            self.lhs.num_vars(),
            self.rhs.num_vars(),
        );

        dbg!(&a_f2_mle);
        dbg!(&beta_g2);

        // result of initializing is the first sumcheck message!
        // --- Basically beta(g_2, p_2) * a_f2(p_2) * a_f3(p_2) ---
        compute_sumcheck_message_copy_phase_mul(&mut [a_f2_mle], &mut beta_g2, 0)
    }

    /// a prove rounds function for testing purposes
    fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<F>, Option<F>)> {
        // initialization
        let first_message = self
            .init_copy_phase(claim.clone())
            .expect("could not evaluate original lhs and rhs");
        dbg!(&first_message);
        let (beta_g2, [a_f2]) = self
            .copy_phase_mles
            .as_mut()
            .ok_or(GateError::CopyPhaseInitError)
            .unwrap();
        let (mut lhs, mut rhs) = (&mut self.lhs.clone(), &mut self.rhs.clone());
        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        let num_rounds_copy_phase = self.new_bits;

        // sumcheck rounds -- over here we bind the copy bits
        // --- At the same time, we're binding the LHS and RHS actual bookkeeping tables over the copy bits ---
        // TODO!(ryancao): Is there a better way we can do that?
        for round in 1..(num_rounds_copy_phase) {
            // challenge = Some(F::from(rng.gen::<u64>()));
            challenge = Some(F::from(2));
            let chal = challenge.unwrap();
            challenges.push(chal);

            let evals =
                prove_round_copy(&mut [a_f2.clone()], &mut lhs, &mut rhs, beta_g2, round, chal).unwrap();
            messages.push((evals, challenge));
        }

        // final challenge of binding the copy bits
        let final_chal = F::one() + F::one();
        // let final_chal = F::from(rng.gen::<u64>());
        challenges.push(final_chal);
        // a_f2.fix_variable(num_rounds_copy_phase - 1, final_chal);
        // a_f3.fix_variable(num_rounds_copy_phase - 1, final_chal);
        beta_g2
            .beta_update(num_rounds_copy_phase - 1, final_chal)
            .unwrap();
        lhs.fix_variable(num_rounds_copy_phase - 1, final_chal);
        rhs.fix_variable(num_rounds_copy_phase - 1, final_chal);

        dbg!(&beta_g2);
        dbg!(&lhs);
        dbg!(&rhs);

        // grab the bound beta value
        debug_assert_eq!(beta_g2.table.bookkeeping_table().len(), 1); // --- Should be fully bound ---
        let beta_g2_val = beta_g2.table.bookkeeping_table()[0];
        let next_claims = (self.g1_challenges.clone().unwrap(), F::zero());

        // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
        // TODO!(ryancao): Can we get rid of the clones here somehow (for `lhs` and `rhs`)?
        let reduced_gate: MulGate<F, Tr> = MulGate::new(
            self.layer_id.clone(),
            self.nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            self.new_bits,
            Some(beta_g2_val)
        );
        self.reduced_gate = Some(reduced_gate);
        let mut next_messages = self
            .reduced_gate
            .as_mut()
            .unwrap()
            .dummy_prove_rounds(next_claims, rng)
            .unwrap();
        let (next_first, _) = &mut next_messages[0];

        // mutate the message from the non-batched case to include the last challenge from copy phase
        next_messages[0] = (next_first.clone(), Some(final_chal));

        // scale it by the bound beta value
        let scaled_next_messages: Vec<(Vec<F>, Option<F>)> = next_messages
            .into_iter()
            .map(|(evals, chal)| {
                let scaled_evals = evals.into_iter().map(|eval| eval * beta_g2_val).collect_vec();
                (scaled_evals, chal)
            })
            .collect();

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
            } else if (..(num_u + 1)).contains(&i) {
                first_u_challenges.push(chal);
            } else {
                last_v_challenges.push(chal);
            }
        }

        // last challenge
        // let final_chal = F::from(rng.gen::<u64>());
        let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);
        // fix the y variable in the reduced gate at this point too
        fix_var_gate(
            self
                .reduced_gate
                .as_mut()
                .unwrap()
                .phase_2_mles
                .as_mut()
                .unwrap(),
            num_v - 1,
            final_chal,
        );

        // compute the gate function evaluated at bound variables
        let beta_u = BetaTable::new((first_u_challenges.clone(), F::zero())).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), F::zero())).unwrap();
        let beta_g = BetaTable::new((self.g1_challenges.clone().unwrap(), F::zero())).unwrap();
        let f_1_uv =
            self.nonzero_gates
                .clone()
                .into_iter()
                .fold(F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                    let gz = *beta_g
                        .table
                        .bookkeeping_table()
                        .get(z_ind)
                        .unwrap_or(&F::zero());
                    let ux = *beta_u
                        .table
                        .bookkeeping_table()
                        .get(x_ind)
                        .unwrap_or(&F::zero());
                    let vy = *beta_v
                        .table
                        .bookkeeping_table()
                        .get(y_ind)
                        .unwrap_or(&F::zero());
                    acc + gz * ux * vy
                });

        // honestly just checking if get_claims() computes correctly, use this to get the bound f_2 and f_3 values
        let claims = self.get_claims().unwrap().clone();
        let [(_, (_, f2_bound)), (_, (_, f3_bound))] = claims.as_slice() else { return Err(VerifyError::SumcheckBad) };

        let beta_bound = compute_beta_over_two_challenges(
            &self.g2_challenges.clone().unwrap(),
            &first_copy_challenges,
        );

        // evaluate the gate expression
        let final_result = beta_bound * (f_1_uv * (*f2_bound * *f3_bound));

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

/// does all the necessary updates when proving a round for batched gate mles
fn prove_round_copy<F: FieldExt>(
    phase_mles: &mut [DenseMleRef<F>],
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    beta: &mut BetaTable<F>,
    round_index: usize,
    challenge: F,
) -> Result<Vec<F>, GateError> {
    fix_var_gate(phase_mles, round_index - 1, challenge);
    beta.beta_update(round_index - 1, challenge).unwrap();
    // need to separately update these because the phase_lhs and phase_rhs has no version of them
    lhs.fix_variable(round_index - 1, challenge);
    rhs.fix_variable(round_index - 1, challenge);
    compute_sumcheck_message_copy_phase_mul(phase_mles, beta, round_index)
}

/// compute sumcheck message without a beta table!!!!!!!!!!!!!!
fn compute_sumcheck_message_copy_phase_mul<F: FieldExt>(
    mles: &[impl MleRef<F = F>],
    beta: &mut BetaTable<F>,
    round_index: usize,
) -> Result<Vec<F>, GateError> {
    let degree = 2;

    // --- Go through all of the MLEs being multiplied together on the LHS and see if any of them contain an IV ---
    // TODO!(ryancao): Should this not always be true...?
    let independent_variable = mles
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let eval = evaluate_mle_ref_product(mles, independent_variable, degree, beta.table.clone()).unwrap();

    let Evals(evaluations) = eval;

    Ok(evaluations)
}


#[cfg(test)]
mod test {
    use crate::mle::{dense::DenseMle, batched_mulgate::MulGateBatched};
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonTranscript, Transcript};

    use super::*;
    use ark_std::test_rng;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    #[test]
    /// batched test
    fn test_sumcheck_batched_1() {
        let mut rng = test_rng();

        let claim = (
            vec![
                Fr::from(1),
                Fr::from(1),
                Fr::from(1),
                Fr::from(0),
                Fr::from(0),
            ],
            Fr::from(0),
        );
        let new_bits = 1;
        let nonzero_gates = vec![(1, 1, 1)];

        let lhs_v = vec![Fr::from(0), Fr::from(5), Fr::from(1), Fr::from(2)];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

        let rhs_v = vec![Fr::from(0), Fr::from(5), Fr::from(51395810), Fr::from(2)];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut batched_gate_mle: MulGateBatched<Fr, PoseidonTranscript<Fr>> = MulGateBatched::new(
            new_bits,
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            LayerId::Layer(0),
        );
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_2() {
        let mut rng = test_rng();
        let new_bits = 2;

        let claim = (vec![Fr::from(1), Fr::from(1), Fr::from(1)], Fr::from(1));
        let nonzero_gates = vec![(1, 1, 1), (0, 0, 1)];

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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut batched_gate_mle: MulGateBatched<Fr, PoseidonTranscript<Fr>> = MulGateBatched::new(
            new_bits,
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            LayerId::Layer(0),
        );
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_3() {
        let mut rng = test_rng();
        let new_bits = 1;

        let claim = (vec![Fr::from(1), Fr::from(1), Fr::from(1)], Fr::from(24));
        let nonzero_gates = vec![(3, 1, 1)];

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
        // 1, 6
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

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
        // 3, 4
        // 6 * 0
        // 48 * 1
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut batched_gate_mle: MulGateBatched<Fr, PoseidonTranscript<Fr>> = MulGateBatched::new(
            new_bits,
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            LayerId::Layer(0),
        );
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_4() {
        let mut rng = test_rng();
        let new_bits = 2;

        let claim = (
            vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(0)],
            Fr::from(24),
        );
        let nonzero_gates = vec![(3, 1, 1), (2, 1, 0), (1, 0, 1)];

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
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut batched_gate_mle: MulGateBatched<Fr, PoseidonTranscript<Fr>> = MulGateBatched::new(
            new_bits,
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            LayerId::Layer(0),
        );
        let messages_1 = batched_gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = batched_gate_mle.dummy_verify_rounds(messages_1, &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

}
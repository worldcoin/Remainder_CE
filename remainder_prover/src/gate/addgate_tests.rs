use ark_std::rand::Rng;
use itertools::Itertools;
use rayon::prelude::ParallelIterator;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    layer::{claims::Claim, Layer, LayerId},
    mle::beta::BetaTable,
    sumcheck::*,
};
use remainder_shared_types::{transcript::Transcript, FieldExt};

use super::gate_helpers::{
    check_fully_bound, compute_sumcheck_message_add_gate, compute_sumcheck_message_copy_add,
    fix_var_gate, index_mle_indices_gate, prove_round_add, prove_round_copy, GateError,
};
use crate::mle::{
    beta::compute_beta_over_two_challenges,
    dense::{DenseMle, DenseMleRef},
    MleRef,
};

/// very (not) cool addgate
///
/// ## Members
/// * `layer_id` - The GKR layer which this gate refers to
/// * `nonzero_gates` - List of tuples (z, x, y) which refer to a "real" addition gate
/// * `lhs` - f_2(x)
/// * `rhs` - f_3(y)
/// * `beta_g` - \beta(g_1, p_2)
/// * `phase_1_mles` - the mles needed to compute the sumcheck evals for phase 1
/// * `phase_2_mles` - the mles needed to compute the sumcheck evals for phase 2
/// * `num_copy_bits` - length of `p_2`
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct AddGateTest<F: FieldExt, Tr: Transcript<F>> {
    pub layer_id: LayerId,
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    pub lhs_num_vars: usize,
    pub rhs_num_vars: usize,
    pub lhs: DenseMleRef<F>,
    pub rhs: DenseMleRef<F>,
    beta_g: Option<BetaTable<F>>,
    pub phase_1_mles: Option<([DenseMleRef<F>; 2], [DenseMleRef<F>; 1])>,
    pub phase_2_mles: Option<([DenseMleRef<F>; 1], [DenseMleRef<F>; 2])>,
    pub num_dataparallel_bits: usize,
    pub beta_scaled: Option<F>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> AddGateTest<F, Tr> {
    /// new addgate mle (wrapper constructor)
    pub fn new(
        layer_id: LayerId,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: usize,
        beta_scaled: Option<F>,
    ) -> AddGateTest<F, Tr> {
        AddGateTest {
            layer_id,
            nonzero_gates,
            lhs_num_vars: lhs.num_vars(),
            rhs_num_vars: rhs.num_vars(),
            lhs,
            rhs,
            beta_g: None,
            phase_1_mles: None,
            phase_2_mles: None,
            num_dataparallel_bits,
            beta_scaled,
            _marker: PhantomData,
        }
    }

    fn set_beta_g(&mut self, beta_g: BetaTable<F>) {
        self.beta_g = Some(beta_g);
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
    ///
    /// The equation for this overall should be something like:
    /// * V_i(g) = \sum_{x, y} f_1(g, x, y) (f_2(x) + f_3(y))
    /// * V_i(g) = \sum_{x} f_2(x) \sum_{y} f_1(g, x, y) + \sum_{x} (1) \sum_{y} f_1(g, x, y) f_3(y)
    ///
    /// Thus the bookkeeping tables we require are:
    /// * f_2(x)
    /// * f_1'(x) = \sum_{y} f_1(g, x, y)
    /// * h_g(x) = \sum_{y} f_1(g, x, y) f_3(y)
    ///
    /// We are binding `x` first, i.e. in this phase. Thus we compute
    /// two bookkeeping tables,
    /// * f_1'(x) = \sum_{y} f_1(g, x, y) = \sum_{y, z} \beta(g, z) f_1(z, x, y)
    /// * h_g(x) = \sum_{y} f_1(g, x, y) f_3(y) = \sum_{y, z} \beta(g, z) f_1(z, x, y) f_3(y)
    ///
    pub fn init_phase_1(&mut self, claim: Claim<F>) -> Result<Vec<F>, GateError> {
        // --- First compute the bookkeeping table for \beta(g, z) \in \{0, 1\}^{s_i} ---
        let beta_g = if !claim.get_point().is_empty() {
            BetaTable::new(claim.get_point().clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        self.set_beta_g(beta_g.clone());

        // we start indexing at the number of copy bits, because once you get to non-batched add gate, those should be bound
        self.lhs.index_mle_indices(self.num_dataparallel_bits);
        let num_x = self.lhs.num_vars();

        // bookkeeping tables according to libra, set everything to zero for now (we are summing over y so size is 2^(num_x))
        // --- `a_hg_lhs` is f_1'(x) ---
        let mut a_hg_lhs = vec![F::zero(); 1 << num_x];
        // --- `a_hg_rhs` is h_g(x) ---
        let mut a_hg_rhs = vec![F::zero(); 1 << num_x];

        // use the gate function to populate the table using beta table
        // --- Formula is as follows ---
        // f_1'(x) = \sum_{(z, x, y) \in N_x} \beta(g, z)
        // h_g(x) = \sum_{(z, x, y) \in N_x} \beta(g, z) * f_3(y)
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                // TODO!(ryancao): Why do we have an `unwrap_or()` here? Would we ever be out-of-bounds (in either case)?
                let beta_g_at_z = *beta_g
                    .table
                    .bookkeeping_table()
                    .get(z_ind)
                    .unwrap_or(&F::zero());
                let f_3_at_y = *self
                    .rhs
                    .bookkeeping_table()
                    .get(y_ind)
                    .unwrap_or(&F::zero());
                a_hg_lhs[x_ind] += beta_g_at_z;
                a_hg_rhs[x_ind] += beta_g_at_z * f_3_at_y;
            });

        // --- We need to multiply f_1'(x) by f_2(x) ---
        let mut phase_1_lhs = [
            DenseMle::new_from_raw(a_hg_lhs, LayerId::Input(0), None).mle_ref(),
            self.lhs.clone(),
        ];
        // --- The RHS will just be h_g(x), which doesn't get multiplied by f_2(x) ---
        let mut phase_1_rhs = [DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0), None).mle_ref()];
        index_mle_indices_gate(phase_1_lhs.as_mut(), self.num_dataparallel_bits);
        index_mle_indices_gate(phase_1_rhs.as_mut(), self.num_dataparallel_bits);
        self.set_phase_1((phase_1_lhs.clone(), phase_1_rhs.clone()));

        // returns the first sumcheck message
        compute_sumcheck_message_add_gate(&phase_1_lhs, &phase_1_rhs, self.num_dataparallel_bits)
    }

    /// initialize bookkeeping tables for phase 2 of sumcheck
    ///
    /// The equation for this should be something of the form
    /// * \sum_{y} f_1(g, u, y) (f_2(u) + f_3(y))
    /// * = f_2(u) (\sum_{y} f_1(g, u, y) + \sum_{y} f_1(g, u, y) * f_3(y)
    ///
    /// Thus the bookkeeping tables we need in this phase are simply
    /// * `a_f1_lhs`(y) = f_1(g, u, y) = \sum_{x, y, z \in N_x} \beta(g, z) \beta(u, x) f_1(z, x, y)
    /// * `a_f1_rhs`(y) = f_1(g, u, y) * f_3(y) = \sum_{x, y, z \in N_x} \beta(g, z) \beta(u, x) f_1(z, x, y) f_3(y)
    ///
    /// which is not precisely what is implemented below; instead, we multiply
    /// f_2(u) to the lhs bookkeeping table and f_3(y) to the rhs -- actually
    /// that is correct.
    pub fn init_phase_2(&mut self, u_claim: Claim<F>, f_at_u: F) -> Result<Vec<F>, GateError> {
        let beta_g = self
            .beta_g
            .as_ref()
            .expect("beta table should be initialized by now");

        // create a beta table according to the challenges used to bind the x variables
        let beta_u = BetaTable::new(u_claim.get_point().clone()).unwrap();
        let num_y = self.rhs.num_vars();

        // bookkeeping table where we now bind y, so size is 2^(num_y)
        let mut a_f1_lhs = vec![F::zero(); 1 << num_y];
        let mut a_f1_rhs = vec![F::zero(); 1 << num_y];

        // TODO!(ryancao): Potential optimization here -- if the nonzero gates are sorted by `y_ind`, we can
        // parallelize over them as long as we split the parallelism along a change in `y_ind`, since each
        // thread will only ever access its own independent subset of `a_f1_lhs` and `a_f1_rhs`.

        // populate the bookkeeping table
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                // TODO!(ryancao): Should we ever need `unwrap_or`?
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
                let adder = gz * ux;
                a_f1_lhs[y_ind] += adder * f_at_u;
                a_f1_rhs[y_ind] += adder;
            });

        // --- LHS bookkeeping table is already multiplied by f_2(u) ---
        let mut phase_2_lhs = [DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0), None).mle_ref()];
        // --- RHS bookkeeping table needs to be multiplied by f_3(y) ---
        let mut phase_2_rhs = [
            DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0), None).mle_ref(),
            self.rhs.clone(),
        ];
        index_mle_indices_gate(phase_2_lhs.as_mut(), self.num_dataparallel_bits);
        index_mle_indices_gate(phase_2_rhs.as_mut(), self.num_dataparallel_bits);
        self.set_phase_2((phase_2_lhs.clone(), phase_2_rhs.clone()));

        // return the first sumcheck message of this phase
        compute_sumcheck_message_add_gate(&phase_2_lhs, &phase_2_rhs, self.num_dataparallel_bits)
    }

    /// dummy sumcheck prover for this, testing purposes
    fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Result<Vec<(Vec<F>, Option<F>)>, GateError> {
        // initialization
        let first_message = self
            .init_phase_1(claim)
            .expect("could not evaluate original lhs and rhs");
        let (phase_1_lhs, phase_1_rhs) = self
            .phase_1_mles
            .as_mut()
            .ok_or(GateError::Phase1InitError)
            .unwrap();

        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        // number of variables in f2 (or the left sum) is how many rounds are in phase 1
        let num_rounds_phase1 = self.lhs.num_vars();

        // sumcheck rounds (binding x)
        for round in 1..(num_rounds_phase1) {
            challenge = Some(F::from(rng.gen::<u64>()));
            //challenge = Some(F::one());
            let chal = challenge.unwrap();
            challenges.push(chal);
            let eval = prove_round_add(
                round + self.num_dataparallel_bits,
                chal,
                phase_1_lhs,
                phase_1_rhs,
            )
            .unwrap();
            messages.push((eval, challenge));
        }

        // do the final binding of phase 1
        let final_chal = F::from(rng.gen::<u64>());
        //let final_chal = F::from(2_u64);
        challenges.push(final_chal);
        fix_var_gate(
            phase_1_lhs,
            num_rounds_phase1 - 1 + self.num_dataparallel_bits,
            final_chal,
        );
        fix_var_gate(
            phase_1_rhs,
            num_rounds_phase1 - 1 + self.num_dataparallel_bits,
            final_chal,
        );
        let f_2 = &phase_1_lhs[1];

        if f_2.bookkeeping_table.len() == 1 {
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = Claim::new_raw(challenges.clone(), F::zero());

            // first message of the next phase includes the random challenge from the last phase
            // (this transition checks that we did the bookkeeping optimization correctly between each phase)
            let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();
            let (phase_2_lhs, phase_2_rhs) = self
                .phase_2_mles
                .as_mut()
                .ok_or(GateError::Phase2InitError)
                .unwrap();
            messages.push((first_message, Some(final_chal)));

            // number of rounds in phase 2 is number of variables in the right sum, binding y
            let num_rounds_phase2 = self.rhs.num_vars();

            // sumcheck rounds (binding y)
            for round in 1..num_rounds_phase2 {
                challenge = Some(F::from(rng.gen::<u64>()));
                //challenge = Some(F::one());
                let chal = challenge.unwrap();
                challenges.push(chal);
                let eval = prove_round_add(
                    round + self.num_dataparallel_bits,
                    chal,
                    phase_2_lhs,
                    phase_2_rhs,
                )
                .unwrap();
                messages.push((eval, challenge));
            }

            Ok(messages)
        } else {
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
        if claimed_val != claim.get_result() {
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

            if (1..(num_u + 1)).contains(&i) {
                first_u_challenges.push(chal);
            } else {
                last_v_challenges.push(chal);
            }
        }

        let final_chal = F::from(rng.gen::<u64>());
        //let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        // also bind the final challenge (this is in phase 2) to f3 (the right sum)
        let ([_, lhs], _) = self.phase_1_mles.as_mut().unwrap();
        let (_, [_, rhs]) = self.phase_2_mles.as_mut().unwrap();
        rhs.fix_variable(num_v - 1 + self.num_dataparallel_bits, final_chal);
        let f_2_u = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();
        let f_3_v = check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap();

        // evaluate the gate function at the bound points!!!
        let beta_u = BetaTable::new(first_u_challenges.clone()).unwrap();
        let beta_v = BetaTable::new(last_v_challenges.clone()).unwrap();
        let beta_g = self.beta_g.as_ref().unwrap();
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
pub struct AddGateBatchedTest<F: FieldExt, Tr: Transcript<F>> {
    pub new_bits: usize,
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    pub lhs: DenseMleRef<F>,
    pub rhs: DenseMleRef<F>,
    pub num_vars_l: Option<usize>,
    pub num_vars_r: Option<usize>,
    copy_phase_mles: Option<(BetaTable<F>, [DenseMleRef<F>; 2])>,
    pub g1_challenges: Option<Vec<F>>,
    pub g2_challenges: Option<Vec<F>>,
    pub layer_id: LayerId,
    pub reduced_gate: Option<AddGateTest<F, Tr>>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> AddGateBatchedTest<F, Tr> {
    /// new batched addgate thingy
    pub fn new(
        new_bits: usize,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        layer_id: LayerId,
    ) -> Self {
        AddGateBatchedTest {
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
        mles: (BetaTable<F>, [DenseMleRef<F>; 2]),
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
        let challenges = claim.get_point();
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
        let mut beta_g2 = BetaTable::new(g2_challenges.clone()).unwrap();
        let beta_g1 = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };

        // the bookkeeping tables of this phase must have size 2^copy_bits (refer to vibe check B))
        let num_copy_vars = 1 << self.new_bits;
        let mut a_f2 = vec![F::zero(); num_copy_vars];
        let mut a_f3 = vec![F::zero(); num_copy_vars];

        // populate the bookkeeping tables
        // TODO!(ryancao): Good optimization here is to parallelize -- I don't think there are any race conditions
        (0..num_copy_vars).for_each(|idx| {
            let mut adder_f2 = F::zero();
            let mut adder_f3 = F::zero();
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
                    adder_f2 += gz * f2_val;
                    adder_f3 += gz * f3_val;
                });
            a_f2[idx] += adder_f2;
            a_f3[idx] += adder_f3;
        });

        // --- Wrappers over the bookkeeping tables ---
        let mut a_f2_mle = DenseMle::new_from_raw(a_f2, LayerId::Input(0), None).mle_ref();
        a_f2_mle.index_mle_indices(0);
        let mut a_f3_mle = DenseMle::new_from_raw(a_f3, LayerId::Input(0), None).mle_ref();
        a_f3_mle.index_mle_indices(0);
        beta_g2.table.index_mle_indices(0);

        // index original bookkeeping tables to send over to the non-batched add gate after the copy phase
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // --- Sets self internal state ---
        self.set_copy_phase(
            (beta_g2.clone(), [a_f2_mle.clone(), a_f3_mle.clone()]),
            g1_challenges,
            g2_challenges,
            self.lhs.num_vars(),
            self.rhs.num_vars(),
        );

        // result of initializing is the first sumcheck message!
        // --- Basically beta(g_2, p_2) * a_f2(p_2) * a_f3(p_2) ---
        compute_sumcheck_message_copy_add(&mut beta_g2, &mut a_f2_mle, &mut a_f3_mle, 0)
    }

    /// a prove rounds function for testing purposes
    fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Vec<(Vec<F>, Option<F>)> {
        // initialization
        let first_message = self
            .init_copy_phase(claim)
            .expect("could not evaluate original lhs and rhs");
        let (beta_g, [a_f2, a_f3]) = self
            .copy_phase_mles
            .as_mut()
            .ok_or(GateError::CopyPhaseInitError)
            .unwrap();
        let (lhs, rhs) = (&mut self.lhs.clone(), &mut self.rhs.clone());
        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;
        messages.push((first_message, challenge));
        let num_rounds_copy_phase = self.new_bits;

        // sumcheck rounds -- over here we bind the copy bits
        // --- At the same time, we're binding the LHS and RHS actual bookkeeping tables over the copy bits ---
        // TODO!(ryancao): Is there a better way we can do that?
        for round in 1..(num_rounds_copy_phase) {
            challenge = Some(F::from(rng.gen::<u64>()));
            let chal = challenge.unwrap();
            challenges.push(chal);

            let evals = prove_round_copy(a_f2, a_f3, lhs, rhs, beta_g, round, chal).unwrap();
            messages.push((evals, challenge));
        }

        // final challenge of binding the copy bits
        //let final_chal = F::one();
        let final_chal = F::from(rng.gen::<u64>());
        challenges.push(final_chal);
        a_f2.fix_variable(num_rounds_copy_phase - 1, final_chal);
        a_f3.fix_variable(num_rounds_copy_phase - 1, final_chal);
        beta_g
            .beta_update(num_rounds_copy_phase - 1, final_chal)
            .unwrap();
        lhs.fix_variable(num_rounds_copy_phase - 1, final_chal);
        rhs.fix_variable(num_rounds_copy_phase - 1, final_chal);

        // grab the bound beta value
        debug_assert_eq!(beta_g.table.bookkeeping_table().len(), 1); // --- Should be fully bound ---
        let beta_g2 = beta_g.table.bookkeeping_table()[0];
        let next_claims = Claim::new_raw(self.g1_challenges.clone().unwrap(), F::zero());

        // reduced gate is how we represent the rest of the protocol as a non-batched gate mle
        // TODO!(ryancao): Can we get rid of the clones here somehow (for `lhs` and `rhs`)?
        let reduced_gate: AddGateTest<F, Tr> = AddGateTest::new(
            self.layer_id,
            self.nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            self.new_bits,
            Some(beta_g2),
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
                let scaled_evals = evals.into_iter().map(|eval| eval * beta_g2).collect_vec();
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
        if claimed_val != claim.get_result() {
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
        let final_chal = F::from(rng.gen::<u64>());
        //let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);
        // fix the y variable in the reduced gate at this point too
        fix_var_gate(
            &mut self
                .reduced_gate
                .as_mut()
                .unwrap()
                .phase_2_mles
                .as_mut()
                .unwrap()
                .1,
            num_v - 1,
            final_chal,
        );

        // compute the gate function evaluated at bound variables
        let beta_u = BetaTable::new(first_u_challenges.clone()).unwrap();
        let beta_v = BetaTable::new(last_v_challenges.clone()).unwrap();
        let beta_g = BetaTable::new(self.g1_challenges.clone().unwrap()).unwrap();
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
        let ([_, lhs_reduced], _) = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_1_mles
            .as_ref()
            .unwrap()
            .clone();

        let (_, [_, rhs_reduced]) = self
            .reduced_gate
            .as_ref()
            .unwrap()
            .phase_2_mles
            .as_ref()
            .unwrap()
            .clone();

        let (f2_bound, f3_bound) = (
            lhs_reduced.bookkeeping_table[0],
            rhs_reduced.bookkeeping_table[0],
        );

        let beta_bound = compute_beta_over_two_challenges(
            &self.g2_challenges.clone().unwrap(),
            &first_copy_challenges,
        );

        // evaluate the gate expression
        let final_result = beta_bound * (f_1_uv * (f2_bound + f3_bound));

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

#[cfg(test)]
mod test {
    use crate::mle::dense::DenseMle;
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonTranscript, Transcript};

    use super::*;
    use ark_std::test_rng;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    #[test]
    /// non-batched test
    fn test_sumcheck_1() {
        let mut rng = test_rng();

        let claim = Claim::new_raw(vec![Fr::from(1), Fr::from(0), Fr::from(0)], Fr::from(4));
        let nonzero_gates = vec![(1, 1, 1)];

        let lhs_v = vec![Fr::from(1), Fr::from(2)];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

        let rhs_v = vec![Fr::from(51395810), Fr::from(2)];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut gate_mle: AddGateTest<Fr, PoseidonTranscript<Fr>> = AddGateTest::new(
            LayerId::Layer(0),
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            0,
            None,
        );
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// non-batched test
    fn test_sumcheck_2() {
        let mut rng = test_rng();

        let claim = Claim::new_raw(vec![Fr::from(0), Fr::from(1), Fr::from(0)], Fr::from(2));
        let nonzero_gates = vec![(1, 0, 1), (3, 0, 2), (2, 3, 4)];

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

        let mut gate_mle: AddGateTest<Fr, PoseidonTranscript<Fr>> = AddGateTest::new(
            LayerId::Layer(0),
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            0,
            None,
        );
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// non-batched test
    fn test_sumcheck_3() {
        let mut rng = test_rng();

        let claim = Claim::new_raw(vec![Fr::from(1), Fr::from(1), Fr::from(0)], Fr::from(5200));
        let nonzero_gates = vec![(1, 0, 15), (3, 0, 2), (5, 3, 14), (2, 3, 4)];

        let lhs_v = vec![
            Fr::from(19051).neg(),
            Fr::from(119084),
            Fr::from(857911),
            Fr::from(1),
            Fr::from(189571),
            Fr::from(16781),
            Fr::from(75361),
            Fr::from(91901).neg(),
        ];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

        let rhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(24251),
            Fr::from(87591),
            Fr::from(1),
            Fr::from(772751),
            Fr::from(131899).neg(),
            Fr::from(191),
            Fr::from(80951),
            Fr::from(51),
            Fr::from(2).neg(),
            Fr::from(2),
            Fr::from(1),
            Fr::from(3),
            Fr::from(7),
            Fr::from(9999),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut gate_mle: AddGateTest<Fr, PoseidonTranscript<Fr>> = AddGateTest::new(
            LayerId::Layer(0),
            nonzero_gates,
            lhs_mle_ref,
            rhs_mle_ref,
            0,
            None,
        );
        let messages_1 = gate_mle.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res_1 = gate_mle.dummy_verify_rounds(messages_1.unwrap(), &mut rng, claim);
        assert!(verify_res_1.is_ok());
    }

    #[test]
    /// batched test
    fn test_sumcheck_batched_1() {
        let mut rng = test_rng();

        let claim = Claim::new_raw(
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

        let mut batched_gate_mle: AddGateBatchedTest<Fr, PoseidonTranscript<Fr>> =
            AddGateBatchedTest::new(
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

        let claim = Claim::new_raw(vec![Fr::from(1), Fr::from(1), Fr::from(1)], Fr::from(2));
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

        let mut batched_gate_mle: AddGateBatchedTest<Fr, PoseidonTranscript<Fr>> =
            AddGateBatchedTest::new(
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

        let claim = Claim::new_raw(vec![Fr::from(3), Fr::from(1), Fr::from(1)], Fr::from(22));
        let nonzero_gates = vec![(3, 1, 1), (2, 1, 0)];

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
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut batched_gate_mle: AddGateBatchedTest<Fr, PoseidonTranscript<Fr>> =
            AddGateBatchedTest::new(
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

        let claim = Claim::new_raw(
            vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(2)],
            Fr::from(4),
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

        let mut batched_gate_mle: AddGateBatchedTest<Fr, PoseidonTranscript<Fr>> =
            AddGateBatchedTest::new(
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

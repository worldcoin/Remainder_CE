use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    layer::{
        claims::Claim,
        claims::{get_num_wlx_evaluations, ClaimError},
        layer_enum::LayerEnum,
        Layer, LayerBuilder, LayerError, LayerId, VerificationError,
    },
    mle::{beta::BetaTable, mle_enum::MleEnum},
    prover::{SumcheckProof, ENABLE_OPTIMIZATION},
    sumcheck::*,
};
use remainder_shared_types::{transcript::Transcript, FieldExt};

use crate::mle::{
    dense::{DenseMle, DenseMleRef},
    MleRef,
};
use thiserror::Error;

use super::gate_helpers::{
    check_fully_bound, compute_full_gate, compute_sumcheck_message_mul_gate, fix_var_gate,
    index_mle_indices_gate, prove_round_mul, GateError,
};

/// implement the layer trait for addgate struct
impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for MulGate<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let first_message = self
            .init_phase_1(claim)
            .expect("could not evaluate original lhs and rhs")
            .into_iter()
            .map(|eval| eval * self.beta_scaled.unwrap_or(F::one()))
            .collect_vec();
        let phase_1_mles = self
            .phase_1_mles
            .as_mut()
            .ok_or(GateError::Phase1InitError)
            .unwrap();

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
                let eval =
                    prove_round_mul(round + self.num_dataparallel_bits, challenge, phase_1_mles)
                        .unwrap()
                        .into_iter()
                        .map(|eval| eval * self.beta_scaled.unwrap_or(F::one()))
                        .collect_vec();
                transcript
                    .append_field_elements("Sumcheck evaluations", &eval)
                    .unwrap();
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // final challenge after binding x (left side of the sum)
        let final_chal_u = transcript
            .get_challenge("Final Sumcheck challenge for binding x")
            .unwrap();
        challenges.push(final_chal_u);

        fix_var_gate(
            phase_1_mles,
            num_rounds_phase1 - 1 + self.num_dataparallel_bits,
            final_chal_u,
        );

        let f_2 = &phase_1_mles[1];
        if f_2.bookkeeping_table.len() == 1 {
            // first message of the next phase includes the random challenge from the last phase
            // (this transition checks that we did the bookkeeping optimization correctly between each phase)
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = Claim::new_raw(challenges.clone(), F::zero());
            let first_message = self
                .init_phase_2(u_challenges, f_at_u)
                .expect("could not evaluate original lhs and rhs")
                .into_iter()
                .map(|eval| eval * self.beta_scaled.unwrap_or(F::one()))
                .collect_vec();

            if self.rhs_num_vars > 0 {
                let phase_2_mles = self
                    .phase_2_mles
                    .as_mut()
                    .ok_or(GateError::Phase2InitError)
                    .unwrap();

                transcript
                    .append_field_elements("Initial Sumcheck evaluations", &first_message)
                    .unwrap();

                // number of rounds in phase 2 is number of variables in the right sum, binding y
                let num_rounds_phase2 = self.rhs.num_vars();

                // bind y, the right side of the sum
                let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
                    .chain((1..num_rounds_phase2).map(|round| {
                        let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                        challenges.push(challenge);
                        let eval = prove_round_mul(
                            round + self.num_dataparallel_bits,
                            challenge,
                            phase_2_mles,
                        )
                        .unwrap()
                        .into_iter()
                        .map(|eval| eval * self.beta_scaled.unwrap_or(F::one()))
                        .collect_vec();
                        transcript
                            .append_field_elements("Sumcheck evaluations", &eval)
                            .unwrap();
                        Ok::<_, LayerError>(eval)
                    }))
                    .try_collect()?;

                let final_chal = transcript
                    .get_challenge("Final Sumcheck challenge")
                    .unwrap();
                challenges.push(final_chal);
                fix_var_gate(
                    phase_2_mles,
                    num_rounds_phase2 - 1 + self.num_dataparallel_bits,
                    final_chal,
                );

                sumcheck_rounds.extend(sumcheck_rounds_y.into_iter());
            }

            Ok(sumcheck_rounds.into())
        } else {
            Err(LayerError::LayerNotReady)
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

        // first round check
        let claimed_claim = prev_evals[0] + prev_evals[1];
        if claimed_claim != claim.get_result() {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();

        for (i, curr_evals) in sumcheck_rounds.iter().enumerate().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();

            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            transcript
                .append_field_elements("Sumcheck evaluations", curr_evals)
                .unwrap();

            prev_evals = curr_evals;
            challenges.push(challenge);
            if (1..(num_u + 1)).contains(&i) {
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

        if self.rhs_num_vars == 0 {
            first_u_challenges.push(final_chal);
        } else {
            last_v_challenges.push(final_chal);
        }

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        let [_, lhs] = self.phase_1_mles.as_mut().unwrap();
        let [_, rhs] = self.phase_2_mles.as_mut().unwrap();
        let bound_lhs = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();

        let bound_rhs = {
            if self.rhs_num_vars > 0 {
                check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap()
            } else {
                debug_assert_eq!(rhs.bookkeeping_table.len(), 1);
                rhs.bookkeeping_table[0]
            }
        };

        // compute the sum over all the variables of the gate function
        let beta_u = BetaTable::new(first_u_challenges.clone()).unwrap();
        let beta_v = if !last_v_challenges.is_empty() {
            BetaTable::new(last_v_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        let beta_g = if !claim.get_point().is_empty() {
            BetaTable::new(claim.get_point().clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
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

        // get the fully evaluated "expression"
        let fully_evaluated = f_1_uv * (bound_lhs * bound_rhs);
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
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        let mut claims = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];

        // check the left side of the sum (f2(u)) against the challenges made to bind that variable
        if let Some([_, f_2_u]) = &self.phase_1_mles {
            for index in f_2_u.mle_indices() {
                fixed_mle_indices_u.push(
                    index
                        .val()
                        .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
                );
            }
            let val = f_2_u.bookkeeping_table()[0];
            let claim: Claim<F> = Claim::new(
                fixed_mle_indices_u,
                val,
                Some(self.id().clone()),
                Some(f_2_u.get_layer_id()),
                Some(MleEnum::Dense(f_2_u.clone())),
            );
            claims.push(claim);
        } else {
            return Err(LayerError::LayerNotReady);
        }

        let mut fixed_mle_indices_v: Vec<F> = vec![];

        // check the right side of the sum (f3(v)) against the challenges made to bind that variable
        if let Some([_, f_3_v]) = &self.phase_2_mles {
            for index in f_3_v.mle_indices() {
                fixed_mle_indices_v.push(
                    index
                        .val()
                        .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
                );
            }
            let val = f_3_v.bookkeeping_table()[0];
            let claim: Claim<F> = Claim::new(
                fixed_mle_indices_v,
                val,
                Some(self.id().clone()),
                Some(f_3_v.get_layer_id()),
                Some(MleEnum::Dense(f_3_v.clone())),
            );
            claims.push(claim);
        } else {
            return Err(LayerError::LayerNotReady);
        }
        Ok(claims)
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(_builder: L, _id: LayerId) -> Self {
        todo!()
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let num_vars = std::cmp::max(self.lhs.num_vars(), self.rhs.num_vars());
        let (num_evals, _,) = get_num_wlx_evaluations(claim_vecs);

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = (num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();

                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                compute_full_gate(
                    new_chal,
                    &mut self.lhs.clone(),
                    &mut self.rhs.clone(),
                    &self.nonzero_gates,
                    0,
                )
            })
            .collect();

        // concat this with the first k evaluations from the claims to get
        // num_evals evaluations
        let mut claimed_vals = claimed_vals.clone();

        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals;
        Ok(wlx_evals)
    }

    fn get_enum(self) -> crate::layer::layer_enum::LayerEnum<F, Self::Transcript> {
        LayerEnum::MulGate(self)
    }
}

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
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MulGate<F: FieldExt, Tr: Transcript<F>> {
    pub layer_id: LayerId,
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    pub lhs_num_vars: usize,
    pub rhs_num_vars: usize,
    pub lhs: DenseMleRef<F>,
    pub rhs: DenseMleRef<F>,
    beta_g: Option<BetaTable<F>>,
    pub phase_1_mles: Option<[DenseMleRef<F>; 2]>,
    pub phase_2_mles: Option<[DenseMleRef<F>; 2]>,
    pub num_dataparallel_bits: usize,
    pub beta_scaled: Option<F>,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> MulGate<F, Tr> {
    /// new addgate mle (wrapper constructor)
    pub fn new(
        layer_id: LayerId,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: usize,
        beta_scaled: Option<F>,
    ) -> MulGate<F, Tr> {
        MulGate {
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
    fn set_phase_1(&mut self, mles: [DenseMleRef<F>; 2]) {
        self.phase_1_mles = Some(mles);
    }

    /// bookkeeping tables necessary for binding y
    fn set_phase_2(&mut self, mles: [DenseMleRef<F>; 2]) {
        self.phase_2_mles = Some(mles);
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
                a_hg_rhs[x_ind] += beta_g_at_z * f_3_at_y;
            });

        // --- We need to multiply h_g(x) by f_2(x) ---
        let mut phase_1_mles = [
            DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0), None).mle_ref(),
            self.lhs.clone(),
        ];
        index_mle_indices_gate(phase_1_mles.as_mut(), self.num_dataparallel_bits);
        self.set_phase_1(phase_1_mles.clone());

        // returns the first sumcheck message
        compute_sumcheck_message_mul_gate(&phase_1_mles, self.num_dataparallel_bits)
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
        let mut a_f1 = vec![F::zero(); 1 << num_y];

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
                a_f1[y_ind] += adder * f_at_u;
            });

        // --- RHS bookkeeping table needs to be multiplied by f_3(y) ---
        let mut phase_2_mles = [
            DenseMle::new_from_raw(a_f1, LayerId::Input(0), None).mle_ref(),
            self.rhs.clone(),
        ];
        index_mle_indices_gate(phase_2_mles.as_mut(), self.num_dataparallel_bits);
        self.set_phase_2(phase_2_mles.clone());

        // return the first sumcheck message of this phase

        compute_sumcheck_message_mul_gate(&phase_2_mles, self.num_dataparallel_bits)
    }
}

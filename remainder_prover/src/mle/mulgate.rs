use ark_std::{cfg_into_iter, rand::Rng};
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    expression::ExpressionStandard,
    layer::{
        claims::ClaimError, layer_enum::LayerEnum, Claim, Layer, LayerBuilder, LayerError, LayerId,
        VerificationError,
    },
    mle::beta::BetaTable,
    prover::SumcheckProof,
    sumcheck::*,
};
use remainder_shared_types::{transcript::Transcript, FieldExt};

use super::{
    beta::compute_beta_over_two_challenges,
    dense::{DenseMle, DenseMleRef},
    MleIndex, MleRef,
};
use thiserror::Error;

/// implement the layer trait for addgate struct
impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for MulGate<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {

        let first_message = self
            .init_phase_1(claim.clone())
            .expect("could not evaluate original lhs and rhs");
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
                let eval = prove_round(
                    round + self.num_copy_bits,
                    challenge,
                    phase_1_mles
                )
                .unwrap().into_iter().map(|eval| {
                    eval * self.beta_scaled.unwrap_or(F::one())
                }).collect_vec();
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
            num_rounds_phase1 - 1 + self.num_copy_bits,
            final_chal_u,
        );



        let f_2 = &phase_1_mles[1];
        if f_2.bookkeeping_table.len() == 1 {
            // first message of the next phase includes the random challenge from the last phase
            // (this transition checks that we did the bookkeeping optimization correctly between each phase)
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = (challenges.clone(), F::zero());
            let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();

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
                        let eval = prove_round(
                            round + self.num_copy_bits,
                            challenge,
                            phase_2_mles,
                        )
                        .unwrap().into_iter().map(
                            |eval| {
                                eval * self.beta_scaled.unwrap_or(F::one())
                            }
                        ).collect_vec();
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
                    num_rounds_phase2 - 1 + self.num_copy_bits,
                    final_chal,
                );

                sumcheck_rounds.extend(sumcheck_rounds_y.into_iter());

            }

            Ok(sumcheck_rounds.into())
        } 
        
        else {
            dbg!("i hope u dont come here");
            return Err(LayerError::LayerNotReady)
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
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
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
        }
        else {
            last_v_challenges.push(final_chal);
        }

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        let [_, lhs] = self.phase_1_mles.as_mut().unwrap();
        let [_, rhs] = self.phase_2_mles.as_mut().unwrap();
        let bound_lhs = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();
        
        let bound_rhs = {
            if self.rhs_num_vars > 0 {
                check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap()
            }
            else {
                debug_assert_eq!(rhs.bookkeeping_table.len(), 1);
                rhs.bookkeeping_table[0]
            }
        };

        // compute the sum over all the variables of the gate function
        let beta_u = BetaTable::new((first_u_challenges.clone(), bound_lhs)).unwrap();
        let beta_v = if last_v_challenges.len() > 0 {
            BetaTable::new((last_v_challenges.clone(), bound_rhs)).unwrap()
        } else {
            BetaTable {
                layer_claim: (vec![], F::zero()),
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        let beta_g = if claim.0.len() > 0 {
            BetaTable::new((claim.0, F::zero())).unwrap()
        } else {
            BetaTable {
                layer_claim: (vec![], F::zero()),
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        let f_1_uv = self.nonzero_gates.clone().into_iter().fold(
            F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.table.bookkeeping_table().get(z_ind).unwrap_or(&F::zero());
                let ux = *beta_u.table.bookkeeping_table().get(x_ind).unwrap_or(&F::zero());
                let vy = *beta_v.table.bookkeeping_table().get(y_ind).unwrap_or(&F::zero());
                acc + gz * ux * vy
            }
        );

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
    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        let mut claims: Vec<(LayerId, Claim<F>)> = vec![];
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
            claims.push((f_2_u.get_layer_id(), (fixed_mle_indices_u, val)));
        } else {
            dbg!("i really hope u dont come here");
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
            claims.push((f_3_v.get_layer_id(), (fixed_mle_indices_v, val)));
        } else {
            dbg!("i really really hope u dont come here");
            return Err(LayerError::LayerNotReady);
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
                    0,
                );
                eval
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals.clone();
        Ok(wlx_evals)
    }

    fn get_enum(self) -> crate::layer::layer_enum::LayerEnum<F, Self::Transcript> {
        LayerEnum::MulGate(self)
    }
}

/// fully evaluates an addgate expression (for both the batched and non-batched case)
pub fn compute_full_mulgate<F: FieldExt>(
    challenges: Vec<F>,
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    nonzero_gates: &Vec<(usize, usize, usize)>,
    copy_bits: usize,
) -> F {
    // split the challenges into which ones are for batched bits, which ones are for others
    let mut copy_chals: Vec<F> = vec![];
    let mut z_chals: Vec<F> = vec![];
    challenges.into_iter().enumerate().for_each(|(idx, chal)| {
        if (0..copy_bits).contains(&idx) {
            copy_chals.push(chal);
        } else {
            z_chals.push(chal);
        }
    });

    // if the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on z
    let beta_g = BetaTable::new((z_chals, F::zero())).unwrap();
    let zero = F::zero();

    // literally summing over everything else (x, y)
    if copy_bits == 0 {
        let sum =
            nonzero_gates
                .clone()
                .into_iter()
                .fold(F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                    let gz = *beta_g
                        .table
                        .bookkeeping_table()
                        .get(z_ind)
                        .unwrap_or(&F::zero());
                    let ux = lhs.bookkeeping_table().get(x_ind).unwrap_or(&zero);
                    let vy = rhs.bookkeeping_table().get(y_ind).unwrap_or(&zero);
                    acc + gz * ux * vy
                });
        sum
    } else {
        let num_copy_idx = 1 << copy_bits;
        // if the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on p2
        let beta_g2 = BetaTable::new((copy_chals, F::zero())).unwrap();
        let sum = {
            // sum over everything else, outer sum being over p2, inner sum over (x, y)
            (0..(1 << num_copy_idx))
                .into_iter()
                .fold(F::zero(), |acc_outer, idx| {
                    let g2 = *beta_g2
                        .table
                        .bookkeeping_table()
                        .get(idx)
                        .unwrap_or(&F::zero());
                    let inner_sum = nonzero_gates.clone().into_iter().fold(
                        F::zero(),
                        |acc, (z_ind, x_ind, y_ind)| {
                            let gz = *beta_g
                                .table
                                .bookkeeping_table()
                                .get(z_ind)
                                .unwrap_or(&F::zero());
                            let ux = lhs
                                .bookkeeping_table()
                                .get(idx + (x_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            let vy = rhs
                                .bookkeeping_table()
                                .get(idx + (y_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            acc + gz * ux * vy
                        },
                    );
                    acc_outer + (g2 * inner_sum)
                })
        };
        sum
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
    #[error("Beta table not indexed")]
    BetaTableNotIndexed,
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
    pub num_copy_bits: usize,
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
        num_copy_bits: usize,
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
            num_copy_bits,
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

        let beta_g = if claim.0.len() > 0 {
            BetaTable::new(claim).unwrap()
        } else {
            BetaTable {
                layer_claim: (vec![], F::zero()),
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
                relevant_indices: vec![],
            }
        };
        self.set_beta_g(beta_g.clone());

        // we start indexing at the number of copy bits, because once you get to non-batched add gate, those should be bound
        self.lhs.index_mle_indices(self.num_copy_bits);
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
                a_hg_rhs[x_ind] = a_hg_rhs[x_ind] + (beta_g_at_z * f_3_at_y);
            });

        // dbg!(&a_hg_rhs);
        // --- We need to multiply h_g(x) by f_2(x) ---
        let mut phase_1_mles = [
            DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0), None).mle_ref(),
            self.lhs.clone(),
        ];
        index_mle_indices_gate(phase_1_mles.as_mut(), self.num_copy_bits);
        self.set_phase_1(phase_1_mles.clone());

        // returns the first sumcheck message
        compute_sumcheck_message_mul_gate(&phase_1_mles, self.num_copy_bits)
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
        let beta_u = BetaTable::new(u_claim).unwrap();
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
                a_f1[y_ind] = a_f1[y_ind] + (adder * f_at_u);
            });

        // --- RHS bookkeeping table needs to be multiplied by f_3(y) ---
        let mut phase_2_mles = [
            DenseMle::new_from_raw(a_f1, LayerId::Input(0), None).mle_ref(),
            self.rhs.clone(),
        ];
        index_mle_indices_gate(phase_2_mles.as_mut(), self.num_copy_bits);
        self.set_phase_2(phase_2_mles.clone());

        // return the first sumcheck message of this phase
        let hello = compute_sumcheck_message_mul_gate(&phase_2_mles, self.num_copy_bits);
        dbg!(&hello);
        hello
    }

    /// dummy sumcheck prover for this, testing purposes
    pub fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Result<Vec<(Vec<F>, Option<F>)>, GateError> {
        // initialization
        let first_message = self
            .init_phase_1(claim)
            .expect("could not evaluate original lhs and rhs");
        let phase_1_mles = self
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
            dbg!("Am here proving rounds in phase 1");
            challenge = Some(F::from(rng.gen::<u64>()));
            // challenge = Some(F::one() + F::one());
            // challenge = Some(F::one());
            let chal = challenge.unwrap();
            challenges.push(chal);
            let eval =
                prove_round(round + self.num_copy_bits, chal, phase_1_mles).unwrap();
            messages.push((eval, challenge));
        }

        // do the final binding of phase 1
        let final_chal = F::from(rng.gen::<u64>());
        // let final_chal = F::from(2_u64);
        // let final_chal = F::one();
        challenges.push(final_chal);
        fix_var_gate(
            phase_1_mles,
            num_rounds_phase1 - 1 + self.num_copy_bits,
            final_chal,
        );
        let f_2 = &phase_1_mles[1];

        if f_2.bookkeeping_table.len() == 1 {
            let f_at_u = f_2.bookkeeping_table[0];
            let u_challenges = (challenges.clone(), F::zero());

            // first message of the next phase includes the random challenge from the last phase
            // (this transition checks that we did the bookkeeping optimization correctly between each phase)
            let first_message = self.init_phase_2(u_challenges, f_at_u).unwrap();
            let phase_2_mles = self
                .phase_2_mles
                .as_mut()
                .ok_or(GateError::Phase2InitError)
                .unwrap();
            messages.push((first_message, Some(final_chal)));

            // number of rounds in phase 2 is number of variables in the right sum, binding y
            let num_rounds_phase2 = self.rhs.num_vars();

            // sumcheck rounds (binding y)
            for round in 1..num_rounds_phase2 {
                // dbg!("Am here proving rounds in phase 2");
                challenge = Some(F::from(rng.gen::<u64>()));
                // challenge = Some(F::one() + F::one());
                // challenge = Some(F::one());
                let chal = challenge.unwrap();
                challenges.push(chal);
                let eval = prove_round(round + self.num_copy_bits, chal, phase_2_mles)
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

            if (1..(num_u + 1)).contains(&i) {
                first_u_challenges.push(chal);
            } else {
                last_v_challenges.push(chal);
            }
        }

        let final_chal = F::from(rng.gen::<u64>());
        // let final_chal = F::one() + F::one();
        // let final_chal = F::one();
        challenges.push(final_chal);
        last_v_challenges.push(final_chal);

        // we mutate the mles in the struct as we bind variables, so we can check whether they were bound correctly
        // also bind the final challenge (this is in phase 2) to f3 (the right sum)
        let [_, lhs] = self.phase_1_mles.as_mut().unwrap();
        let [_, rhs] = self.phase_2_mles.as_mut().unwrap();
        rhs.fix_variable(num_v - 1 + self.num_copy_bits, final_chal);
        let f_2_u = check_fully_bound(&mut [lhs.clone()], first_u_challenges.clone()).unwrap();
        let f_3_v = check_fully_bound(&mut [rhs.clone()], last_v_challenges.clone()).unwrap();

        // evaluate the gate function at the bound points!!!
        let beta_u = BetaTable::new((first_u_challenges.clone(), f_2_u)).unwrap();
        let beta_v = BetaTable::new((last_v_challenges.clone(), f_3_v)).unwrap();
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
        let last_v_bound = f_1_uv * (f_2_u * f_3_v);
        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final round of sumcheck check
        if last_v_bound != prev_at_r {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }
}

/// evaluate_mle_ref_product without beta tables........
///
/// ---
///
/// Given (possibly half-fixed) bookkeeping tables of the MLEs which are multiplied,
/// e.g. V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// computes g_k(x) = \sum_{b_{k + 1}, ..., b_n} V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// at `degree + 1` points.
///
/// ## Arguments
/// * `mle_refs` - MLEs pointing to the actual bookkeeping tables for the above
/// * `independent_variable` - whether the `x` from above resides within at least one of the `mle_refs`
/// * `degree` - degree of `g_k(x)`, i.e. number of evaluations to send (minus one!)
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
        // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
        let eval_count = degree + 1;

        // iterate across all pairs of evaluations
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
                        // iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
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
        let sum = cfg_into_iter!((0..1 << (max_num_vars))).fold(
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
                        mle_ref
                            .bookkeeping_table()
                            .get(index)
                            .cloned()
                            .unwrap_or(F::zero())
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                // --- Combine them into the accumulator ---
                // Note that the accumulator stores g(0), g(1), ..., g(d - 1)
                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = sum.reduce(|| F::zero(), |acc, partial| acc + partial);

        Ok(Evals(vec![sum; degree]))
    }
}

/// checks whether mle was bound correctly to all the challenge points!!!!!!!!!!
pub fn check_fully_bound<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    challenges: Vec<F>,
) -> Result<F, GateError> {
    
    let mles_bound: Vec<bool> = mle_refs
        .iter()
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

            if indices != challenges {
                false
            } else {
                true
            }
        })
        .collect();


    if mles_bound.contains(&false) {
        dbg!(&mle_refs);
        dbg!(&challenges);
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
pub fn index_mle_indices_gate<F: FieldExt>(mle_refs: &mut [impl MleRef<F = F>], index: usize) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        mle_ref.index_mle_indices(index);
    })
}

/// fix variable for an array of mles
pub fn fix_var_gate<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    round_index: usize,
    challenge: F,
) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        if mle_ref
            .mle_indices()
            .contains(&MleIndex::IndexedBit(round_index))
        {
            mle_ref.fix_variable(round_index, challenge);
        }
    })
}

/// compute sumcheck message without a beta table!!!!!!!!!!!!!!
pub fn compute_sumcheck_message_mul_gate<F: FieldExt>(
    mles: &[impl MleRef<F = F>],
    round_index: usize,
) -> Result<Vec<F>, GateError> {
    // for gate mles, degree always 2 for left and right side because on each side we are taking the product of two bookkkeping tables
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
    let eval = evaluate_mle_ref_product_gate(mles, independent_variable, degree).unwrap();

    let Evals(evaluations) = eval;

    Ok(evaluations)
}

/// Computes a round of the sumcheck protocol on this Layer
fn prove_round<F: FieldExt>(
    round_index: usize,
    challenge: F,
    mles: &mut [impl MleRef<F = F>],
) -> Result<Vec<F>, GateError> {
    fix_var_gate(mles, round_index - 1, challenge);
    compute_sumcheck_message_mul_gate(mles, round_index)
}

#[cfg(test)]
mod test {
    use crate::mle::{dense::DenseMle, batched_mulgate::MulGateBatched};
    use remainder_shared_types::transcript::{poseidon_transcript::PoseidonTranscript, Transcript};

    use super::*;
    use ark_std::test_rng;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;

    #[test]
    /// non-batched test
    fn test_sumcheck_1() {
        let mut rng = test_rng();

        let claim = (vec![Fr::from(1), Fr::from(0), Fr::from(0)], Fr::from(4));
        let nonzero_gates = vec![(1, 1, 1)];

        let lhs_v = vec![Fr::from(1), Fr::from(2)];
        let lhs_mle_ref = DenseMle::new_from_raw(lhs_v, LayerId::Input(0), None).mle_ref();

        let rhs_v = vec![Fr::from(51395810), Fr::from(2)];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut gate_mle: MulGate<Fr, PoseidonTranscript<Fr>> = MulGate::new(
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

        let claim = (vec![Fr::from(0), Fr::from(1), Fr::from(0)], Fr::from(15));
        let nonzero_gates = vec![(1, 0, 1), (3, 0, 2), (2, 3, 4)];

        let lhs_v = vec![
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
            Fr::from(3),
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
            Fr::from(5),
            Fr::from(1),
            Fr::from(1),
            Fr::from(1),
        ];
        let rhs_mle_ref = DenseMle::new_from_raw(rhs_v, LayerId::Input(0), None).mle_ref();

        let mut gate_mle: MulGate<Fr, PoseidonTranscript<Fr>> = MulGate::new(
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

        let claim = (vec![Fr::from(1), Fr::from(1), Fr::from(0)], Fr::from(462005801).neg());
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

        let mut gate_mle: MulGate<Fr, PoseidonTranscript<Fr>> = MulGate::new(
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

}

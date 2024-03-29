use std::{cmp::max, marker::PhantomData};

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use remainder_shared_types::{transcript::Transcript, FieldExt};
use serde::{Deserialize, Serialize};

use crate::{
    gate::gate_helpers::{prove_round_dataparallel_phase, prove_round_gate},
    layer::{
        claims::{get_num_wlx_evaluations, Claim, ClaimError},
        layer_enum::LayerEnum,
        Layer, LayerBuilder, LayerError, LayerId, VerificationError,
    },
    mle::{
        beta::{compute_beta_over_two_challenges, BetaTable},
        dense::{DenseMle, DenseMleRef},
        mle_enum::MleEnum,
        MleRef,
    },
    prover::SumcheckProof,
    sumcheck::{self, evaluate_at_a_point, Evals},
};

use super::gate_helpers::{
    check_fully_bound, compute_full_gate, compute_sumcheck_message_no_beta_table,
    index_mle_indices_gate, libra_giraffe, GateError,
};

#[derive(PartialEq, Serialize, Deserialize, Clone, Debug, Copy)]

/// operations that are currently supported by the gate. binary because these are fan-in-two gates
pub enum BinaryOperation {
    /// an add gate
    Add,
    /// a mul gate
    Mul,
}

impl BinaryOperation {
    /// method to perform the respective operation
    pub fn perform_operation<F: FieldExt>(&self, a: F, b: F) -> F {
        match self {
            BinaryOperation::Add => a + b,
            BinaryOperation::Mul => a * b,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
/// generic gate struct -- the binary operation performed by the gate is specified by
/// the `gate_operation` parameter. additionally, the number of dataparallel variables
/// is specified by `num_dataparallel_bits` in order to account for batched and un-batched
/// gates.
pub struct Gate<F: FieldExt, Tr: Transcript<F>> {
    /// the layer id associated with this gate layer
    pub layer_id: LayerId,
    /// the number of bits representing the number of "dataparallel" copies of the circuit
    pub num_dataparallel_bits: usize,
    /// a vector of tuples representing the "nonzero" gates, especially useful in the sparse case
    /// the format is (z, x, y) where the gate at label z is the output of performing an operation
    /// on gates with labels x and y
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    /// the left side of the expression, i.e. the mle that makes up the "x" variables
    pub lhs: DenseMleRef<F>,
    /// the right side of the expression, i.e. the mle that makes up the "y" variables
    pub rhs: DenseMleRef<F>,
    /// the mles that are constructed when initializing phase 1 (binding the x variables)
    pub phase_1_mles: Option<Vec<Vec<DenseMleRef<F>>>>,
    /// the mles that are constructed when initializing phase 2 (binding the y variables)
    pub phase_2_mles: Option<Vec<Vec<DenseMleRef<F>>>>,
    /// the gate operation representing the fan-in-two relationship
    pub gate_operation: BinaryOperation,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for Gate<F, Tr> {
    type Transcript = Tr;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let mut sumcheck_rounds = vec![];
        let (mut beta_g1, mut beta_g2) = self.compute_beta_tables(claim.get_point());
        let mut beta_g2_fully_bound = F::one();
        // we perform the dataparallel initiliazation only if there is at least one variable
        // representing which copy we are in
        if self.num_dataparallel_bits > 0 {
            let (dataparallel_rounds, beta_g2_bound) = self
                .perform_dataparallel_phase(
                    claim.get_point().clone(),
                    &mut beta_g1,
                    &mut beta_g2,
                    transcript,
                )
                .unwrap();
            beta_g2_fully_bound = beta_g2_bound;
            sumcheck_rounds.extend(dataparallel_rounds);
        }
        // we perform the rounds binding "x" variables (phase 1) and the rounds binding "y" variables (phase 2) in sequence
        let (phase_1_rounds, f2_at_u, u_challenges) = self
            .perform_phase_1(
                claim.get_point()[self.num_dataparallel_bits..].to_vec(),
                &beta_g1,
                beta_g2_fully_bound,
                transcript,
            )
            .unwrap();
        let phase_2_rounds = self
            .perform_phase_2(
                f2_at_u,
                u_challenges,
                beta_g1,
                beta_g2_fully_bound,
                transcript,
            )
            .unwrap();
        sumcheck_rounds.extend(phase_1_rounds);
        sumcheck_rounds.extend(phase_2_rounds);

        // the concatenation of all of these rounds is the proof resulting from a gate layer
        Ok(Some(sumcheck_rounds).into())
    }

    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_rounds: Option<Vec<Vec<F>>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError> {
        let sumcheck_rounds = sumcheck_rounds.unwrap();
        let mut prev_evals = &sumcheck_rounds[0];
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let mut first_copy_challenges = vec![];
        let num_u = self.lhs.original_num_vars();

        // first round check against the claim!!!!
        let claimed_val = sumcheck_rounds[0][0] + sumcheck_rounds[0][1];
        if claimed_val != claim.get_result() {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();

        // check each of the messages -- note that here the verifier doesn't actually see the difference
        // between dataparallel rounds, phase 1 rounds, and phase 2 rounds--the prover's proof reads
        // as a single continuous proof.
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

            // we want to separate the challenges into which ones are from the dataprallel bits, which ones
            // are for binding x (phase 1), and which are for binding y (phase 2)
            if (..=self.num_dataparallel_bits).contains(&i) {
                first_copy_challenges.push(challenge);
            } else if (..=num_u).contains(&i) {
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

        // this belongs in the last challenge bound to y
        if self.rhs.num_vars() == 0 {
            first_u_challenges.push(final_chal);
        } else {
            last_v_challenges.push(final_chal);
        }

        // we want to grab the mutated bookkeeping tables from the "reduced_gate", this is the non-batched version
        let lhs_reduced = self.phase_1_mles.clone().unwrap()[0][1].clone();
        let rhs_reduced = self.phase_2_mles.clone().unwrap()[0][1].clone();

        // since the original mles are batched, the challenges are the concat of the copy bits and the variable bound bits
        let lhs_challenges = [
            first_copy_challenges.clone().as_slice(),
            first_u_challenges.clone().as_slice(),
        ]
        .concat();
        let rhs_challenges = [
            first_copy_challenges.clone().as_slice(),
            last_v_challenges.clone().as_slice(),
        ]
        .concat();

        let g2_challenges = claim.get_point()[..self.num_dataparallel_bits].to_vec();
        let g1_challenges = claim.get_point()[self.num_dataparallel_bits..].to_vec();

        // compute the gate function bound at those variables
        // beta table corresponding to the equality of binding the x variables to u
        let beta_u = BetaTable::new(first_u_challenges.clone()).unwrap();
        // beta table corresponding to the equality of binding the y variables to v
        let beta_v = if !last_v_challenges.is_empty() {
            BetaTable::new(last_v_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };
        // beta table representing all "z" label challenges
        let beta_g = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };
        // multiply the corresponding entries of the beta tables to get the full value of the gate function
        // i.e. f1(z, x, y) bound at the challenges f1(g1, u, v)
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

        // check that the original mles have been bound correctly -- this is
        // what we get from the reduced gate
        check_fully_bound(&mut [lhs_reduced.clone()], lhs_challenges).unwrap();
        check_fully_bound(&mut [rhs_reduced.clone()], rhs_challenges).unwrap();
        let f2_bound = lhs_reduced.bookkeeping_table()[0];
        let f3_bound = rhs_reduced.bookkeeping_table()[0];
        let beta_bound = compute_beta_over_two_challenges(&g2_challenges, &first_copy_challenges);

        // compute the final result of the bound expression
        let final_result =
            beta_bound * (f_1_uv * self.gate_operation.perform_operation(f2_bound, f3_bound));

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // final check in sumcheck
        if final_result != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        let lhs_reduced = self.phase_1_mles.clone().unwrap()[0][1].clone();
        let rhs_reduced = self.phase_2_mles.clone().unwrap()[0][1].clone();

        let mut claims = vec![];

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
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_u,
            val,
            Some(self.id().clone()),
            Some(self.lhs.get_layer_id()),
            Some(MleEnum::Dense(lhs_reduced.clone())),
        );
        claims.push(claim);

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
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_v,
            val,
            Some(self.id().clone()),
            Some(self.rhs.get_layer_id()),
            Some(MleEnum::Dense(rhs_reduced.clone())),
        );
        claims.push(claim);

        Ok(claims)
    }

    /// Gets this layer's id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }

    /// Create new ConcreteLayer from a LayerBuilder -- not necessary for a gate layer because we
    /// don't use layer builders in order to construct gate layers (as they don't operate on expressions,
    /// just mles)
    fn new<L: LayerBuilder<F>>(_builder: L, _id: LayerId) -> Self {
        unimplemented!()
    }

    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        _claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let (num_evals, _) = get_num_wlx_evaluations(claim_vecs);

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
                    self.num_dataparallel_bits,
                )
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut claimed_vals = claimed_vals.clone();

        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals;
        Ok(wlx_evals)
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        LayerEnum::Gate(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> Gate<F, Tr> {
    /// Construct a new gate layer
    ///
    /// # Arguments
    /// * `num_dataparallel_bits`: an optional representing the number of bits representing the circuit copy we are looking at. None
    /// if this is not dataparallel, otherwise specify the number of bits
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// * `lhs`: the flattened mle representing the left side of the summation
    /// * `rhs`: the flattened mle representing the right side of the summation
    /// * `gate_operation`: which operation the gate is performing. right now, can either be an 'add' or 'mul' gate
    /// * `layer_id`: the id representing which current layer this is
    ///
    /// # Returns
    /// A `Gate` struct that can now prove and verify rounds
    pub fn new(
        num_dataparallel_bits: Option<usize>,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        gate_operation: BinaryOperation,
        layer_id: LayerId,
    ) -> Self {
        Gate {
            num_dataparallel_bits: num_dataparallel_bits.unwrap_or(0),
            nonzero_gates,
            lhs,
            rhs,
            layer_id,
            phase_1_mles: None,
            phase_2_mles: None,
            gate_operation,
            _marker: PhantomData,
        }
    }

    fn compute_beta_tables(&mut self, challenges: &Vec<F>) -> (BetaTable<F>, BetaTable<F>) {
        let mut g2_challenges = vec![];
        let mut g1_challenges = vec![];

        challenges
            .iter()
            .enumerate()
            .for_each(|(bit_idx, challenge)| {
                if bit_idx < self.num_dataparallel_bits {
                    g2_challenges.push(*challenge);
                } else {
                    g1_challenges.push(*challenge);
                }
            });

        // create two separate beta tables for each, as they are handled differently
        let mut beta_g2 = if !g2_challenges.is_empty() {
            BetaTable::new(g2_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };
        beta_g2.table.index_mle_indices(0);
        let beta_g1 = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };

        (beta_g1, beta_g2)
    }

    /// initialize the dataparallel phase: construct the necessary mles and return the first sumcheck mesage
    /// this will then set the necessary fields of the Gate struct so that the dataparallel bits can be
    /// correctly bound during the first `num_dataparallel_bits` rounds of sumcheck
    fn init_dataparallel_phase(&mut self, challenges: Vec<F>) -> Result<Vec<F>, GateError> {
        let mut g2_challenges: Vec<F> = vec![];
        let mut g1_challenges: Vec<F> = vec![];

        // we split the claim challenges into two -- the first copy_bits number of challenges are referred
        // to as g2, and the rest are referred to as g1. this distinguishes batching from non-batching internally
        challenges
            .iter()
            .enumerate()
            .for_each(|(bit_idx, challenge)| {
                if bit_idx < self.num_dataparallel_bits {
                    g2_challenges.push(*challenge);
                } else {
                    g1_challenges.push(*challenge);
                }
            });

        // create two separate beta tables for each, as they are handled differently
        let mut beta_g2 = BetaTable::new(g2_challenges.clone()).unwrap();
        beta_g2.table.index_mle_indices(0);
        let beta_g1 = if !g1_challenges.is_empty() {
            BetaTable::new(g1_challenges.clone()).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };

        // index original bookkeeping tables to send over to the non-batched mul gate after the copy phase
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // result of initializing is the first sumcheck message!
        let first_sumcheck_message = libra_giraffe(
            &self.lhs,
            &self.rhs,
            &beta_g2.table,
            &beta_g1.table,
            self.gate_operation.clone(),
            &self.nonzero_gates,
            self.num_dataparallel_bits,
        );

        first_sumcheck_message
    }

    /// initialize phase 1, or the necessary mles in order to bind the variables in the `lhs` of the
    /// expression. once this phase is initialized, the sumcheck rounds binding the "x" variables can
    /// be performed
    fn init_phase_1(&mut self, challenges: Vec<F>) -> Result<Vec<F>, GateError> {
        let beta_g1 = if !challenges.is_empty() {
            BetaTable::new(challenges).unwrap()
        } else {
            BetaTable {
                layer_claim_vars: vec![],
                table: DenseMle::new_from_raw(vec![F::one()], LayerId::Input(0), None).mle_ref(),
            }
        };

        self.lhs.index_mle_indices(self.num_dataparallel_bits);
        let num_x = self.lhs.num_vars();

        // because we are binding `x` variables after this phase, all bookkeeping tables should have size
        // 2^(number of x variables)
        let mut a_hg_rhs = vec![F::zero(); 1 << num_x];
        let mut a_hg_lhs = vec![F::zero(); 1 << num_x];

        // over here, we are looping through the nonzero gates using the Libra trick. this takes advantage
        // of the sparsity of the gate function. if we have the following expression:
        // f1(z, x, y)(f2(x) + f3(y)) then because we are only binding the "x" variables, we can simply
        // distribute over the y variables and construct bookkeeping tables that are size 2^(num_x_variables)
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                let beta_g_at_z = *beta_g1
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
                if self.gate_operation == BinaryOperation::Add {
                    a_hg_lhs[x_ind] += beta_g_at_z;
                }
            });

        let a_hg_rhs_mle_ref = DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0), None).mle_ref();

        // the actual mles defer based on whether we are doing a add gate or a mul gate, because
        // in the case of an add gate, we distribute the gate function whereas in the case of the
        // mul gate, we simply take the product over all three mles.
        let mut phase_1_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_hg_lhs, LayerId::Input(0), None).mle_ref(),
                        self.lhs.clone(),
                    ],
                    vec![a_hg_rhs_mle_ref],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![a_hg_rhs_mle_ref, self.lhs.clone()]]
            }
        };

        phase_1_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_bits);
        });

        self.phase_1_mles = Some(phase_1_mles.clone());

        let max_deg = phase_1_mles
            .iter()
            .fold(0, |acc, elem| max(acc, elem.len()));

        let evals_vec = phase_1_mles
            .iter_mut()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_bits, max_deg)
                    .unwrap()
            })
            .collect_vec();
        let final_evals = evals_vec
            .clone()
            .into_iter()
            .skip(1)
            .fold(Evals(evals_vec[0].clone()), |acc, elem| acc + Evals(elem));
        let Evals(final_vec_evals) = final_evals;
        Ok(final_vec_evals)
    }

    /// initialize phase 2, or the necessary mles in order to bind the variables in the `rhs` of the
    /// expression. once this phase is initialized, the sumcheck rounds binding the "y" variables can
    /// be performed
    fn init_phase_2(
        &mut self,
        u_claim: Vec<F>,
        f_at_u: F,
        beta_g1: &BetaTable<F>,
    ) -> Result<Vec<F>, GateError> {
        // create a beta table according to the challenges used to bind the x variables
        let beta_u = BetaTable::new(u_claim).unwrap();
        let num_y = self.rhs.num_vars();

        // because we are binding the "y" variables, the size of the bookkeeping tables after this init
        // phase are 2^(number of y variables)
        let mut a_f1_lhs = vec![F::zero(); 1 << num_y];
        let mut a_f1_rhs = vec![F::zero(); 1 << num_y];

        // by the time we get here, we assume the "x" variables and "dataparallel" variables have been
        // bound. therefore, we are simply scaling by the appropriate gate value and the fully bound
        // `lhs` of the expression in order to compute the necessary mles, once again using the Libra trick
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                let gz = *beta_g1
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
                if self.gate_operation == BinaryOperation::Add {
                    a_f1_rhs[y_ind] += adder;
                }
            });

        let a_f1_lhs_mle_ref = DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0), None).mle_ref();
        // --- We need to multiply h_g(x) by f_2(x) ---
        let mut phase_2_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0), None).mle_ref(),
                        self.rhs.clone(),
                    ],
                    vec![a_f1_lhs_mle_ref],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![a_f1_lhs_mle_ref, self.rhs.clone()]]
            }
        };

        phase_2_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_bits);
        });
        self.phase_2_mles = Some(phase_2_mles.clone());

        // return the first sumcheck message of this phase
        let max_deg = phase_2_mles
            .iter()
            .fold(0, |acc, elem| max(acc, elem.len()));

        let evals_vec = phase_2_mles
            .iter_mut()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_bits, max_deg)
                    .unwrap()
            })
            .collect_vec();
        let final_evals = evals_vec
            .clone()
            .into_iter()
            .skip(1)
            .fold(Evals(evals_vec[0].clone()), |acc, elem| acc + Evals(elem));
        let Evals(final_vec_evals) = final_evals;
        Ok(final_vec_evals)
    }

    // once the initialization of the dataparallel phase is done, we can perform the dataparallel phase.
    // this means that we are binding all bits that represent which copy of the circuit we are in.
    fn perform_dataparallel_phase(
        &mut self,
        claim: Vec<F>,
        beta_g1: &mut BetaTable<F>,
        beta_g2: &mut BetaTable<F>,
        transcript: &mut <Gate<F, Tr> as Layer<F>>::Transcript,
    ) -> Result<(Vec<Vec<F>>, F), LayerError> {
        // initialization, first message comes from here
        let mut challenges: Vec<F> = vec![];

        let first_message = self.init_dataparallel_phase(claim).expect(
            "could not evaluate original lhs and rhs in order to get first sumcheck message",
        );

        let (lhs, rhs) = (&mut self.lhs, &mut self.rhs);

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &first_message)
            .unwrap();
        let num_rounds_copy_phase = self.num_dataparallel_bits;

        // do the first dataparallel bits number sumcheck rounds using libra giraffe
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_copy_phase).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                challenges.push(challenge);
                let eval = prove_round_dataparallel_phase(
                    lhs,
                    rhs,
                    &beta_g1,
                    beta_g2,
                    round,
                    challenge,
                    &self.nonzero_gates,
                    self.num_dataparallel_bits - round,
                    self.gate_operation.clone(),
                )
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
        // fix the variable and everything as you would in the last round of sumcheck
        // the evaluations from this is what you return from the first round of sumcheck in the next phase!
        beta_g2
            .beta_update(num_rounds_copy_phase - 1, final_chal_copy)
            .unwrap();
        self.lhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g2.table.bookkeeping_table().len() == 1 {
            let beta_g2_fully_bound = beta_g2.table.bookkeeping_table()[0];
            Ok((sumcheck_rounds, beta_g2_fully_bound))
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    // we are binding the "x" variables of the `lhs`. at the end of this, the lhs of the expression
    // assuming we have a fan-in-two gate must be fully bound.
    fn perform_phase_1(
        &mut self,
        challenge: Vec<F>,
        beta_g1: &BetaTable<F>,
        beta_g2_fully_bound: F,
        transcript: &mut <Gate<F, Tr> as Layer<F>>::Transcript,
    ) -> Result<(Vec<Vec<F>>, F, Vec<F>), LayerError> {
        let first_message = self
            .init_phase_1(challenge)
            .expect("could not evaluate original lhs and rhs")
            .into_iter()
            .map(|eval| eval * beta_g2_fully_bound)
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
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_phase1).map(|round| {
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                challenges.push(challenge);
                // if there are dataparallel bits, we want to start at that index
                let eval =
                    prove_round_gate(round + self.num_dataparallel_bits, challenge, phase_1_mles)
                        .into_iter()
                        .map(|eval| eval * beta_g2_fully_bound)
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

        phase_1_mles.iter_mut().for_each(|mle_ref_vec| {
            mle_ref_vec.iter_mut().for_each(|mle_ref| {
                mle_ref.fix_variable(
                    num_rounds_phase1 - 1 + self.num_dataparallel_bits,
                    final_chal_u,
                );
            })
        });

        let f_2 = phase_1_mles[0][1].clone();

        if f_2.bookkeeping_table().len() == 1 {
            let f2_at_u = f_2.bookkeeping_table()[0];
            Ok((sumcheck_rounds, f2_at_u, challenges))
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    // these are the rounds binding the "y" variables of the expression. at the end of this, the entire
    // expression is fully bound because this is the last phase in proving the gate layer.
    fn perform_phase_2(
        &mut self,
        f_at_u: F,
        phase_1_challenges: Vec<F>,
        beta_g1: BetaTable<F>,
        beta_g2_fully_bound: F,
        transcript: &mut <Gate<F, Tr> as Layer<F>>::Transcript,
    ) -> Result<Vec<Vec<F>>, LayerError> {
        let first_message = self
            .init_phase_2(phase_1_challenges.clone(), f_at_u, &beta_g1)
            .unwrap()
            .into_iter()
            .map(|eval| eval * beta_g2_fully_bound)
            .collect_vec();

        let mut challenges: Vec<F> = vec![];

        if self.rhs.num_vars() > 0 {
            let phase_2_mles = self
                .phase_2_mles
                .as_mut()
                .ok_or(GateError::Phase2InitError)
                .unwrap();

            transcript
                .append_field_elements("Initial Sumcheck evaluations", &first_message)
                .unwrap();

            let num_rounds_phase2 = self.rhs.num_vars();

            // bind y, the right side of the sum
            let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
                .chain((1..num_rounds_phase2).map(|round| {
                    let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();
                    challenges.push(challenge);
                    let eval = prove_round_gate(
                        round + self.num_dataparallel_bits,
                        challenge,
                        phase_2_mles,
                    )
                    .into_iter()
                    .map(|eval| eval * beta_g2_fully_bound)
                    .collect_vec();
                    transcript
                        .append_field_elements("Sumcheck evaluations", &eval)
                        .unwrap();
                    Ok::<_, LayerError>(eval)
                }))
                .try_collect()?;

            // final round of sumcheck
            let final_chal = transcript
                .get_challenge("Final Sumcheck challenge")
                .unwrap();
            challenges.push(final_chal);

            phase_2_mles.iter_mut().for_each(|mle_ref_vec| {
                mle_ref_vec.iter_mut().for_each(|mle_ref| {
                    mle_ref.fix_variable(
                        num_rounds_phase2 - 1 + self.num_dataparallel_bits,
                        final_chal,
                    );
                })
            });

            Ok(sumcheck_rounds_y)
        } else {
            Ok(vec![])
        }
    }
}

/// For circuit serialization to hash the circuit description into the transcript.
impl<F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> Gate<F, Tr> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct GateCircuitDesc<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>>(
            &'a Gate<F, Tr>,
        );

        impl<'a, F: std::fmt::Debug + FieldExt, Tr: Transcript<F>> std::fmt::Display
            for GateCircuitDesc<'a, F, Tr>
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Gate")
                    .field("lhs_mle_ref_layer_id", &self.0.lhs.get_layer_id())
                    .field("lhs_mle_ref_mle_indices", &self.0.lhs.mle_indices())
                    .field("rhs_mle_ref_layer_id", &self.0.rhs.get_layer_id())
                    .field("rhs_mle_ref_mle_indices", &self.0.rhs.mle_indices())
                    .field("add_nonzero_gates", &self.0.nonzero_gates)
                    .field("num_dataparallel_bits", &self.0.num_dataparallel_bits)
                    .finish()
            }
        }
        GateCircuitDesc(self)
    }
}

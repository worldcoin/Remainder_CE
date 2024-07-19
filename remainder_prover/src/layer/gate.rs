//! module for defining the gate layer, uses the libra trick
//! to reduce the number of rounds for gate layers (with binary operations)

mod gate_helpers;
#[cfg(test)]
mod tests;

use std::cmp::max;

use ark_std::cfg_into_iter;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::{
        circuit_expr::CircuitMle,
        generic_expr::Expression,
        verifier_expr::{VerifierExpr, VerifierMle},
    },
    layer::{Layer, LayerError, LayerId, VerificationError},
    mle::{betavalues::BetaValues, dense::DenseMle, mle_enum::MleEnum, Mle},
    prover::SumcheckProof,
    sumcheck::{evaluate_at_a_point, Evals},
};

use self::gate_helpers::{
    check_fully_bound, compute_full_gate, compute_sumcheck_message_no_beta_table,
    index_mle_indices_gate, libra_giraffe, prove_round_dataparallel_phase, prove_round_gate,
    GateError,
};

use super::{CircuitLayer, VerifierLayer};

#[derive(PartialEq, Serialize, Deserialize, Clone, Debug, Copy)]

/// Operations that are currently supported by the gate. Binary because these
/// are fan-in-two gates.
pub enum BinaryOperation {
    /// An addition gate.
    Add,

    /// A multiplication gate.
    Mul,
}

impl BinaryOperation {
    /// Method to perform the respective operation.
    pub fn perform_operation<F: FieldExt>(&self, a: F, b: F) -> F {
        match self {
            BinaryOperation::Add => a + b,
            BinaryOperation::Mul => a * b,
        }
    }
}

/// Generic gate struct -- the binary operation performed by the gate is specified by
/// the `gate_operation` parameter. Additionally, the number of dataparallel variables
/// is specified by `num_dataparallel_bits` in order to account for batched and un-batched
/// gates.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct Gate<F: FieldExt> {
    /// The layer id associated with this gate layer.
    pub layer_id: LayerId,
    /// The number of bits representing the number of "dataparallel" copies of the circuit.
    pub num_dataparallel_bits: usize,
    /// A vector of tuples representing the "nonzero" gates, especially useful in the sparse case
    /// the format is (z, x, y) where the gate at label z is the output of performing an operation
    /// on gates with labels x and y.
    pub nonzero_gates: Vec<(usize, usize, usize)>,
    /// The left side of the expression, i.e. the mle that makes up the "x" variables.
    pub lhs: DenseMle<F>,
    /// The right side of the expression, i.e. the mle that makes up the "y" variables.
    pub rhs: DenseMle<F>,
    /// The mles that are constructed when initializing phase 1 (binding the x variables).
    pub phase_1_mles: Option<Vec<Vec<DenseMle<F>>>>,
    /// The mles that are constructed when initializing phase 2 (binding the y variables).
    pub phase_2_mles: Option<Vec<Vec<DenseMle<F>>>>,
    /// The gate operation representing the fan-in-two relationship.
    pub gate_operation: BinaryOperation,
}

impl<F: FieldExt> Layer<F> for Gate<F> {
    type CircuitLayer = CircuitGateLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError> {
        let mut sumcheck_rounds = vec![];
        let (mut beta_g1, mut beta_g2) = self.compute_beta_tables(claim.get_point());
        let mut beta_g2_fully_bound = F::ONE;
        // We perform the dataparallel initiliazation only if there is at least one variable
        // representing which copy we are in.
        if self.num_dataparallel_bits > 0 {
            let (dataparallel_rounds, beta_g2_bound) = self
                .perform_dataparallel_phase(
                    claim.get_point().clone(),
                    &mut beta_g1,
                    &mut beta_g2,
                    transcript_writer,
                )
                .unwrap();
            beta_g2_fully_bound = beta_g2_bound;
            sumcheck_rounds.extend(dataparallel_rounds.0);
        }
        // We perform the rounds binding "x" variables (phase 1) and the rounds binding "y" variables (phase 2) in sequence.
        let (phase_1_rounds, f2_at_u, u_challenges) = self
            .perform_phase_1(
                claim.get_point()[self.num_dataparallel_bits..].to_vec(),
                beta_g2_fully_bound,
                transcript_writer,
            )
            .unwrap();
        let phase_2_rounds = self
            .perform_phase_2(
                f2_at_u,
                u_challenges,
                beta_g1,
                beta_g2_fully_bound,
                transcript_writer,
            )
            .unwrap();
        sumcheck_rounds.extend(phase_1_rounds.0);
        sumcheck_rounds.extend(phase_2_rounds.0);

        // The concatenation of all of these rounds is the proof resulting from a gate layer.
        //Ok(sumcheck_rounds.into())
        Ok(())
    }

    fn into_circuit_layer(&self) -> Result<CircuitGateLayer<F>, LayerError> {
        todo!()
    }
}

/// The circuit-description counterpart of a Gate layer description.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct CircuitGateLayer<F: FieldExt> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// The gate operation representing the fan-in-two relationship.
    gate_operation: BinaryOperation,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x, y) where the gate at label z is
    /// the output of performing an operation on gates with labels x and y.
    wiring: Vec<(usize, usize, usize)>,

    /// The left side of the expression, i.e. the mle that makes up the "x"
    /// variables.
    lhs_mle: CircuitMle<F>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: CircuitMle<F>,

    /// The number of bits representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_bits: usize,
}

impl<F: FieldExt> CircuitLayer<F> for CircuitGateLayer<F> {
    type VerifierLayer = VerifierGateLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        todo!()
        /*
        let sumcheck_rounds = sumcheck_rounds.0;
        let mut prev_evals = &sumcheck_rounds[0];
        let mut challenges = vec![];
        let mut first_u_challenges = vec![];
        let mut last_v_challenges = vec![];
        let mut first_copy_challenges = vec![];
        let num_u = self.lhs.original_num_vars();

        // First round check against the claim.
        let claimed_val = sumcheck_rounds[0][0] + sumcheck_rounds[0][1];
        if claimed_val != claim.get_result() {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        let num_elements = sumcheck_rounds[0].len();
        let transcript_sumcheck_round_zero = transcript_reader
            .consume_elements("Initial Sumcheck evaluations", num_elements)
            .map_err(LayerError::TranscriptError)?;
        debug_assert_eq!(transcript_sumcheck_round_zero, sumcheck_rounds[0]);

        // Check each of the messages -- note that here the verifier doesn't actually see the difference
        // between dataparallel rounds, phase 1 rounds, and phase 2 rounds--the prover's proof reads
        // as a single continuous proof.
        for (i, curr_evals) in sumcheck_rounds.iter().enumerate().skip(1) {
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
                .unwrap();
            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            let num_curr_evals = curr_evals.len();
            let transcript_curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", num_curr_evals)
                .map_err(LayerError::TranscriptError)?;
            debug_assert_eq!(transcript_curr_evals, *curr_evals);
            prev_evals = curr_evals;
            challenges.push(challenge);

            // We want to separate the challenges into which ones are from the dataprallel bits, which ones
            // are for binding x (phase 1), and which are for binding y (phase 2).
            if (..=self.num_dataparallel_bits).contains(&i) {
                first_copy_challenges.push(challenge);
            } else if (..=num_u).contains(&i) {
                first_u_challenges.push(challenge);
            } else {
                last_v_challenges.push(challenge);
            }
        }

        // Final round of sumcheck.
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .map_err(LayerError::TranscriptError)?;
        challenges.push(final_chal);

        // This belongs in the last challenge bound to y.
        if self.rhs.num_iterated_vars() == 0 {
            first_u_challenges.push(final_chal);
        } else {
            last_v_challenges.push(final_chal);
        }

        // We want to grab the mutated bookkeeping tables from the "reduced_gate", this is the non-batched version.
        let lhs_reduced = self.phase_1_mles.clone().unwrap()[0][1].clone();
        let rhs_reduced = self.phase_2_mles.clone().unwrap()[0][1].clone();

        // Since the original mles are batched, the challenges are the concat of the copy bits and the variable bound bits.
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

        // Compute the gate function bound at those variables.
        // Beta table corresponding to the equality of binding the x variables to u.
        let beta_u = BetaValues::new_beta_equality_mle(first_u_challenges.clone());
        // Beta table corresponding to the equality of binding the y variables to v.
        let beta_v = BetaValues::new_beta_equality_mle(last_v_challenges.clone());
        // Beta table representing all "z" label challenges.
        let beta_g = BetaValues::new_beta_equality_mle(g1_challenges);
        // Multiply the corresponding entries of the beta tables to get the full value of the gate function
        // i.e. f1(z, x, y) bound at the challenges f1(g1, u, v).
        let f_1_uv =
            self.nonzero_gates
                .clone()
                .into_iter()
                .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                    let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                    let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);
                    let vy = *beta_v.bookkeeping_table().get(y_ind).unwrap_or(&F::ZERO);
                    acc + gz * ux * vy
                });

        // Check that the original mles have been bound correctly -- this is
        // what we get from the reduced gate.
        check_fully_bound(&mut [lhs_reduced.clone()], lhs_challenges).unwrap();
        check_fully_bound(&mut [rhs_reduced.clone()], rhs_challenges).unwrap();
        let f2_bound = lhs_reduced.bookkeeping_table()[0];
        let f3_bound = rhs_reduced.bookkeeping_table()[0];
        let beta_bound =
            BetaValues::compute_beta_over_two_challenges(&g2_challenges, &first_copy_challenges);

        // Compute the final result of the bound expression.
        let final_result =
            beta_bound * (f_1_uv * self.gate_operation.perform_operation(f2_bound, f3_bound));

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        // Final check in sumcheck.
        if final_result != prev_at_r {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
        */
    }
}

/// The verifier's counterpart of a Gate layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierGateLayer<F: FieldExt> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// The gate operation representing the fan-in-two relationship.
    gate_operation: BinaryOperation,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x, y) where the gate at label z is
    /// the output of performing an operation on gates with labels x and y.
    wiring: Vec<(usize, usize, usize)>,

    /// The left side of the expression, i.e. the mle that makes up the "x"
    /// variables.
    lhs_mle: VerifierMle<F>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: VerifierMle<F>,

    /// The number of bits representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_bits: usize,
}

impl<F: FieldExt> VerifierLayer<F> for VerifierGateLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for Gate<F> {
    /// Get the claims that this layer makes on other layers.
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let lhs_reduced = self.phase_1_mles.clone().unwrap()[0][1].clone();
        let rhs_reduced = self.phase_2_mles.clone().unwrap()[0][1].clone();

        let mut claims = vec![];

        // Grab the claim on the left side.
        let mut fixed_mle_indices_u: Vec<F> = vec![];
        for index in lhs_reduced.mle_indices() {
            fixed_mle_indices_u.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = lhs_reduced.bookkeeping_table()[0];
        let claim: ClaimMle<F> = ClaimMle::new(
            fixed_mle_indices_u,
            val,
            Some(self.layer_id()),
            Some(self.lhs.get_layer_id()),
            Some(MleEnum::Dense(lhs_reduced)),
        );
        claims.push(claim);

        // Grab the claim on the right side.
        let mut fixed_mle_indices_v: Vec<F> = vec![];
        for index in rhs_reduced.mle_indices() {
            fixed_mle_indices_v.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = rhs_reduced.bookkeeping_table()[0];
        let claim: ClaimMle<F> = ClaimMle::new(
            fixed_mle_indices_v,
            val,
            Some(self.layer_id()),
            Some(self.rhs.get_layer_id()),
            Some(MleEnum::Dense(rhs_reduced)),
        );
        claims.push(claim);

        Ok(claims)
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for VerifierGateLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        todo!()
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for Gate<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        _claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // Get the number of evaluations.
        let (num_evals, _) = get_num_wlx_evaluations(claim_vecs);

        // We already have the first #claims evaluations, get the next num_evals - #claims evaluations.
        let next_evals: Vec<F> = (num_claims..num_evals)
            .map(|idx| {
                // Get the challenge l(idx).
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(claim_vecs)
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

        // Concat this with the first k evaluations from the claims to get num_evals evaluations.
        let mut claimed_vals = claimed_vals.to_vec();

        claimed_vals.extend(&next_evals);
        let wlx_evals = claimed_vals;
        Ok(wlx_evals)
    }
}

impl<F: FieldExt> Gate<F> {
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
        lhs: DenseMle<F>,
        rhs: DenseMle<F>,
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
        }
    }

    fn compute_beta_tables(&mut self, challenges: &[F]) -> (DenseMle<F>, DenseMle<F>) {
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

        // Create two separate beta tables for each, as they are handled differently.
        let mut beta_g2 = BetaValues::new_beta_equality_mle(g2_challenges);
        beta_g2.index_mle_indices(0);
        let beta_g1 = BetaValues::new_beta_equality_mle(g1_challenges);

        (beta_g1, beta_g2)
    }

    /// Initialize the dataparallel phase: construct the necessary mles and return the first sumcheck message.
    /// This will then set the necessary fields of the [Gate] struct so that the dataparallel bits can be
    /// correctly bound during the first `num_dataparallel_bits` rounds of sumcheck.
    fn init_dataparallel_phase(&mut self, challenges: Vec<F>) -> Result<Vec<F>, GateError> {
        let mut g2_challenges: Vec<F> = vec![];
        let mut g1_challenges: Vec<F> = vec![];

        // We split the claim challenges into two -- the first copy_bits number of challenges are referred
        // to as g2, and the rest are referred to as g1. This distinguishes batching from non-batching internally.
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

        // Create two separate beta tables for each, as they are handled differently.
        let mut beta_g2 = BetaValues::new_beta_equality_mle(g2_challenges);
        beta_g2.index_mle_indices(0);
        let beta_g1 = BetaValues::new_beta_equality_mle(g1_challenges);

        // Index original bookkeeping tables to send over to the non-batched mul gate after the copy phase.
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // Result of initializing is the first sumcheck message.

        libra_giraffe(
            &self.lhs,
            &self.rhs,
            &beta_g2,
            &beta_g1,
            self.gate_operation,
            &self.nonzero_gates,
            self.num_dataparallel_bits,
        )
    }

    /// Initialize phase 1, or the necessary mles in order to bind the variables in the `lhs` of the
    /// expression. Once this phase is initialized, the sumcheck rounds binding the "x" variables can
    /// be performed.
    fn init_phase_1(&mut self, challenges: Vec<F>) -> Result<Vec<F>, GateError> {
        let beta_g1 = BetaValues::new_beta_equality_mle(challenges);

        self.lhs.index_mle_indices(self.num_dataparallel_bits);
        let num_x = self.lhs.num_iterated_vars();

        // Because we are binding `x` variables after this phase, all bookkeeping tables should have size
        // 2^(number of x variables).
        let mut a_hg_rhs = vec![F::ZERO; 1 << num_x];
        let mut a_hg_lhs = vec![F::ZERO; 1 << num_x];

        // Over here, we are looping through the nonzero gates using the Libra trick. This takes advantage
        // of the sparsity of the gate function. if we have the following expression:
        // f1(z, x, y)(f2(x) + f3(y)) then because we are only binding the "x" variables, we can simply
        // distribute over the y variables and construct bookkeeping tables that are size 2^(num_x_variables).
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                let beta_g_at_z = *beta_g1.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let f_3_at_y = *self.rhs.bookkeeping_table().get(y_ind).unwrap_or(&F::ZERO);
                a_hg_rhs[x_ind] += beta_g_at_z * f_3_at_y;
                if self.gate_operation == BinaryOperation::Add {
                    a_hg_lhs[x_ind] += beta_g_at_z;
                }
            });

        let a_hg_rhs_mle_ref = DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0));

        // The actual mles defer based on whether we are doing a add gate or a mul gate, because
        // in the case of an add gate, we distribute the gate function whereas in the case of the
        // mul gate, we simply take the product over all three mles.
        let mut phase_1_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_hg_lhs, LayerId::Input(0)),
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

    /// Initialize phase 2, or the necessary mles in order to bind the variables in the `rhs` of the
    /// expression. Once this phase is initialized, the sumcheck rounds binding the "y" variables can
    /// be performed.
    fn init_phase_2(
        &mut self,
        u_claim: Vec<F>,
        f_at_u: F,
        beta_g1: &DenseMle<F>,
    ) -> Result<Vec<F>, GateError> {
        // Create a beta table according to the challenges used to bind the x variables.
        let beta_u = BetaValues::new_beta_equality_mle(u_claim);
        let num_y = self.rhs.num_iterated_vars();

        // Because we are binding the "y" variables, the size of the bookkeeping tables after this init
        // phase are 2^(number of y variables).
        let mut a_f1_lhs = vec![F::ZERO; 1 << num_y];
        let mut a_f1_rhs = vec![F::ZERO; 1 << num_y];

        // By the time we get here, we assume the "x" variables and "dataparallel" variables have been
        // bound. Therefore, we are simply scaling by the appropriate gate value and the fully bound
        // `lhs` of the expression in order to compute the necessary mles, once again using the Libra trick
        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind, y_ind)| {
                let gz = *beta_g1.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);
                let adder = gz * ux;
                a_f1_lhs[y_ind] += adder * f_at_u;
                if self.gate_operation == BinaryOperation::Add {
                    a_f1_rhs[y_ind] += adder;
                }
            });

        let a_f1_lhs_mle_ref = DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0));
        // --- We need to multiply h_g(x) by f_2(x) ---
        let mut phase_2_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0)),
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

        // Return the first sumcheck message of this phase.
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

    // Once the initialization of the dataparallel phase is done, we can perform the dataparallel phase.
    // This means that we are binding all bits that represent which copy of the circuit we are in.
    fn perform_dataparallel_phase(
        &mut self,
        claim: Vec<F>,
        beta_g1: &mut DenseMle<F>,
        beta_g2: &mut DenseMle<F>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(SumcheckProof<F>, F), LayerError> {
        // Initialization, first message comes from here.
        let mut challenges: Vec<F> = vec![];

        let first_message = self.init_dataparallel_phase(claim).expect(
            "could not evaluate original lhs and rhs in order to get first sumcheck message",
        );

        let (lhs, rhs) = (&mut self.lhs, &mut self.rhs);

        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);
        let num_rounds_copy_phase = self.num_dataparallel_bits;

        // Do the first dataparallel bits number sumcheck rounds using libra giraffe.
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_copy_phase).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                let eval = prove_round_dataparallel_phase(
                    lhs,
                    rhs,
                    beta_g1,
                    beta_g2,
                    round,
                    challenge,
                    &self.nonzero_gates,
                    self.num_dataparallel_bits - round,
                    self.gate_operation,
                )
                .unwrap();
                transcript_writer.append_elements("Sumcheck evaluations", &eval);
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // Bind the final challenge, update the final beta table.
        let final_chal_copy = transcript_writer.get_challenge("Final Sumcheck challenge");
        // Fix the variable and everything as you would in the last round of sumcheck
        // the evaluations from this is what you return from the first round of sumcheck in the next phase!
        beta_g2.fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.lhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g2.bookkeeping_table().len() == 1 {
            let beta_g2_fully_bound = beta_g2.bookkeeping_table()[0];
            Ok((sumcheck_rounds.into(), beta_g2_fully_bound))
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    // We are binding the "x" variables of the `lhs`. At the end of this, the lhs of the expression
    // assuming we have a fan-in-two gate must be fully bound.
    fn perform_phase_1(
        &mut self,
        challenge: Vec<F>,
        beta_g2_fully_bound: F,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(SumcheckProof<F>, F, Vec<F>), LayerError> {
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
        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);
        let num_rounds_phase1 = self.lhs.num_iterated_vars();

        // Sumcheck rounds (binding x).
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_phase1).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                // If there are dataparallel bits, we want to start at that index.
                let eval =
                    prove_round_gate(round + self.num_dataparallel_bits, challenge, phase_1_mles)
                        .into_iter()
                        .map(|eval| eval * beta_g2_fully_bound)
                        .collect_vec();
                transcript_writer.append_elements("Sumcheck evaluations", &eval);
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // Final challenge after binding x (left side of the sum).
        let final_chal_u =
            transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
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
            Ok((sumcheck_rounds.into(), f2_at_u, challenges))
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    // These are the rounds binding the "y" variables of the expression. At the end of this, the entire
    // expression is fully bound because this is the last phase in proving the gate layer.
    fn perform_phase_2(
        &mut self,
        f_at_u: F,
        phase_1_challenges: Vec<F>,
        beta_g1: DenseMle<F>,
        beta_g2_fully_bound: F,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let first_message = self
            .init_phase_2(phase_1_challenges, f_at_u, &beta_g1)
            .unwrap()
            .into_iter()
            .map(|eval| eval * beta_g2_fully_bound)
            .collect_vec();

        let mut challenges: Vec<F> = vec![];

        if self.rhs.num_iterated_vars() > 0 {
            let phase_2_mles = self
                .phase_2_mles
                .as_mut()
                .ok_or(GateError::Phase2InitError)
                .unwrap();

            transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);

            let num_rounds_phase2 = self.rhs.num_iterated_vars();

            // Bind y, the right side of the sum.
            let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
                .chain((1..num_rounds_phase2).map(|round| {
                    let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                    challenges.push(challenge);
                    let eval = prove_round_gate(
                        round + self.num_dataparallel_bits,
                        challenge,
                        phase_2_mles,
                    )
                    .into_iter()
                    .map(|eval| eval * beta_g2_fully_bound)
                    .collect_vec();
                    transcript_writer.append_elements("Sumcheck evaluations", &eval);
                    Ok::<_, LayerError>(eval)
                }))
                .try_collect()?;

            // Final round of sumcheck.
            let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge");
            challenges.push(final_chal);

            phase_2_mles.iter_mut().for_each(|mle_ref_vec| {
                mle_ref_vec.iter_mut().for_each(|mle_ref| {
                    mle_ref.fix_variable(
                        num_rounds_phase2 - 1 + self.num_dataparallel_bits,
                        final_chal,
                    );
                })
            });

            Ok(sumcheck_rounds_y.into())
        } else {
            Ok(vec![].into())
        }
    }
}

/// For circuit serialization to hash the circuit description into the transcript.
impl<F: std::fmt::Debug + FieldExt> Gate<F> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct GateCircuitDesc<'a, F: std::fmt::Debug + FieldExt>(&'a Gate<F>);

        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for GateCircuitDesc<'a, F> {
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

//! module for defining the gate layer, uses the libra trick
//! to reduce the number of rounds for gate layers (with binary operations)

/// Helper functions used in the gate sumcheck algorithms.
pub mod gate_helpers;
mod new_interface_tests;

use std::{cmp::max, collections::HashSet};

use ark_std::cfg_into_iter;
use gate_helpers::bind_round_gate;
use itertools::Itertools;
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::verifier_expr::VerifierMle,
    layer::{Layer, LayerError, LayerId, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, Mle, MleIndex,
    },
    prover::SumcheckProof,
    sumcheck::{evaluate_at_a_point, SumcheckEvals},
};

pub use self::gate_helpers::{
    check_fully_bound, compute_full_gate, compute_sumcheck_message_gate,
    compute_sumcheck_message_no_beta_table, compute_sumcheck_messages_data_parallel_gate,
    index_mle_indices_gate, prove_round_dataparallel_phase, GateError,
};

use super::{
    layer_enum::{LayerEnum, VerifierLayerEnum},
    LayerDescription, VerifierLayer,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
    pub fn perform_operation<F: Field>(&self, a: F, b: F) -> F {
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
#[serde(bound = "F: Field")]
pub struct GateLayer<F: Field> {
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
    /// Temp for debugging
    challenges: Vec<F>,
}

impl<F: Field> Layer<F> for GateLayer<F> {
    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        let mut sumcheck_rounds = vec![];
        let (mut beta_g1, mut beta_g2) = self.compute_beta_tables(claim.get_point());
        let mut beta_g2_fully_bound = F::ONE;
        // We perform the dataparallel initialization only if there is at least one variable
        // representing which copy we are in.
        if self.num_dataparallel_bits > 0 {
            let (dataparallel_rounds, beta_g2_bound) = self
                .perform_dataparallel_phase(&mut beta_g1, &mut beta_g2, transcript_writer)
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

        // --- Finally, send the claimed values for each of the bound MLEs to the verifier ---
        // First, send the claimed value of V_{i + 1}(g_2, u)
        let lhs_reduced = &self.phase_1_mles.as_ref().unwrap()[0][1];
        let rhs_reduced = &self.phase_2_mles.as_ref().unwrap()[0][1];
        debug_assert!(lhs_reduced.bookkeeping_table().len() == 1);
        transcript_writer.append(
            "Evaluation of V_{i + 1}(g_2, u)",
            lhs_reduced.bookkeeping_table()[0],
        );
        // Next, send the claimed value of V_{i + 1}(g_2, v)
        debug_assert!(rhs_reduced.bookkeeping_table().len() == 1);
        transcript_writer.append(
            "Evaluation of V_{i + 1}(g_2, v)",
            rhs_reduced.bookkeeping_table()[0],
        );

        Ok(())
    }

    fn initialize_sumcheck(&mut self, _claim_point: &[F]) -> Result<(), LayerError> {
        todo!()
    }

    fn compute_round_sumcheck_message(&self, _round_index: usize) -> Result<Vec<F>, LayerError> {
        todo!()
    }

    fn bind_round_variable(
        &mut self,
        _round_index: usize,
        _challenge: F,
    ) -> Result<(), LayerError> {
        todo!()
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        todo!()
    }

    fn max_degree(&self) -> usize {
        todo!()
    }

    fn get_post_sumcheck_layer(
        &self,
        _round_challenges: &[F],
        _claim_challenges: &[F],
    ) -> super::product::PostSumcheckLayer<F, F> {
        todo!()
    }
}

/// The circuit-description counterpart of a Gate layer description.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct GateLayerDescription<F: Field> {
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
    lhs_mle: MleDescription<F>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: MleDescription<F>,

    /// The number of bits representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_bits: usize,
}

impl<F: Field> GateLayerDescription<F> {
    /// Constructor for a [GateLayerDescription].
    pub fn new(
        num_dataparallel_bits: Option<usize>,
        wiring: Vec<(usize, usize, usize)>,
        lhs_circuit_mle: MleDescription<F>,
        rhs_circuit_mle: MleDescription<F>,
        gate_layer_id: LayerId,
        gate_operation: BinaryOperation,
    ) -> Self {
        GateLayerDescription {
            id: gate_layer_id,
            gate_operation,
            wiring,
            lhs_mle: lhs_circuit_mle,
            rhs_mle: rhs_circuit_mle,
            num_dataparallel_bits: num_dataparallel_bits.unwrap_or(0),
        }
    }
}

/// Degree of independent variable is cubic for mul dataparallel binding and
/// quadratic for all other bindings (see below expression to verify for yourself!)
///
/// V_i(g_2, g_1) = \sum_{p_2, x, y} \beta(g_2, p_2) f_1(g_1, x, y) (V_{i + 1}(p_2, x) \op V_{i + 1}(p_2, y))
const DATAPARALLEL_ROUND_MUL_NUM_EVALS: usize = 4;
const DATAPARALLEL_ROUND_ADD_NUM_EVALS: usize = 3;
const NON_DATAPARALLEL_ROUND_MUL_NUM_EVALS: usize = 3;
const NON_DATAPARALLEL_ROUND_ADD_NUM_EVALS: usize = 3;

impl<F: Field> LayerDescription<F> for GateLayerDescription<F> {
    type VerifierLayer = VerifierGateLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        // --- Storing challenges for the sake of claim generation later ---
        let mut challenges = vec![];

        // --- WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL ---
        // --- INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED ---
        // --- BITS ---
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;

        // --- Store all prover sumcheck messages to check against ---
        let mut sumcheck_messages: Vec<Vec<F>> = vec![];

        // First round check against the claim.
        let first_round_num_evals = match (self.gate_operation, self.num_dataparallel_bits) {
            (BinaryOperation::Add, 0) => NON_DATAPARALLEL_ROUND_ADD_NUM_EVALS,
            (BinaryOperation::Mul, 0) => NON_DATAPARALLEL_ROUND_MUL_NUM_EVALS,
            (BinaryOperation::Add, _) => DATAPARALLEL_ROUND_ADD_NUM_EVALS,
            (BinaryOperation::Mul, _) => DATAPARALLEL_ROUND_MUL_NUM_EVALS,
        };
        let first_round_sumcheck_messages = transcript_reader
            .consume_elements("Initial Sumcheck evaluations", first_round_num_evals)?;
        sumcheck_messages.push(first_round_sumcheck_messages.clone());

        // Check: V_i(g_2, g_1) =? g_1(0) + g_1(1)
        // TODO(ryancao): SUPER overloaded notation (in e.g. above comments); fix across the board
        if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1] != claim.get_result()
        {
            return Err(VerificationError::SumcheckStartFailed);
        }

        // Check each of the messages -- note that here the verifier doesn't actually see the difference
        // between dataparallel rounds, phase 1 rounds, and phase 2 rounds; instead, the prover's proof reads
        // as a single continuous proof.
        for sumcheck_round_idx in 1..self.num_dataparallel_bits + num_u + num_v {
            // --- Read challenge r_{i - 1} from transcript ---
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
                .unwrap();
            let g_i_minus_1_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();

            // --- Evaluate g_{i - 1}(r_{i - 1}) ---
            let prev_at_r = evaluate_at_a_point(&g_i_minus_1_evals, challenge).unwrap();

            // --- Read off g_i(0), g_i(1), ..., g_i(d) from transcript ---
            let univariate_num_evals = match (
                sumcheck_round_idx < self.num_dataparallel_bits, // 0-indexed, so strictly less-than is correct
                self.gate_operation,
            ) {
                (true, BinaryOperation::Add) => DATAPARALLEL_ROUND_ADD_NUM_EVALS,
                (true, BinaryOperation::Mul) => DATAPARALLEL_ROUND_MUL_NUM_EVALS,
                (false, BinaryOperation::Add) => NON_DATAPARALLEL_ROUND_ADD_NUM_EVALS,
                (false, BinaryOperation::Mul) => NON_DATAPARALLEL_ROUND_MUL_NUM_EVALS,
            };

            let curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", univariate_num_evals)
                .unwrap();

            // --- Check: g_i(0) + g_i(1) =? g_{i - 1}(r_{i - 1}) ---
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerificationError::SumcheckFailed);
            };

            // --- Add the prover message to the sumcheck messages ---
            sumcheck_messages.push(curr_evals);
            // Add the challenge.
            challenges.push(challenge);
        }

        // Final round of sumcheck -- sample r_n from transcript.
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        // --- Create the resulting verifier layer for claim tracking ---
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_gate_layer = self
            .convert_into_verifier_layer(&challenges, claim.get_point(), transcript_reader)
            .unwrap();
        let final_result = verifier_gate_layer.evaluate(&claim);

        // Finally, compute g_n(r_n).
        let g_n_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();
        let prev_at_r = evaluate_at_a_point(&g_n_evals, final_chal).unwrap();

        // Final check in sumcheck.
        if final_result != prev_at_r {
            return Err(VerificationError::FinalSumcheckFailed);
        }

        Ok(VerifierLayerEnum::Gate(verifier_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;
        (0..num_u + num_v + self.num_dataparallel_bits).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        // --- WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL ---
        // --- INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED ---
        // --- BITS ---
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_bits;

        // We want to separate the challenges into which ones are from the dataparallel bits, which ones
        // are for binding x (phase 1), and which are for binding y (phase 2).
        let mut sumcheck_bindings_vec = sumcheck_bindings.to_vec();
        let last_v_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_bits + num_u);
        let first_u_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_bits);
        let dataparallel_challenges = sumcheck_bindings_vec;

        assert_eq!(last_v_challenges.len(), num_v);

        // Since the original mles are dataparallel, the challenges are the concat of the copy bits and the variable bound bits.
        let lhs_challenges = dataparallel_challenges
            .iter()
            .chain(first_u_challenges.iter())
            .copied()
            .collect_vec();
        let rhs_challenges = dataparallel_challenges
            .iter()
            .chain(last_v_challenges.iter())
            .copied()
            .collect_vec();

        let lhs_verifier_mle = self
            .lhs_mle
            .into_verifier_mle(&lhs_challenges, transcript_reader)
            .unwrap();
        let rhs_verifier_mle = self
            .rhs_mle
            .into_verifier_mle(&rhs_challenges, transcript_reader)
            .unwrap();

        // --- Create the resulting verifier layer for claim tracking ---
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_gate_layer = VerifierGateLayer {
            layer_id: self.layer_id(),
            gate_operation: self.gate_operation,
            wiring: self.wiring.clone(),
            lhs_mle: lhs_verifier_mle,
            rhs_mle: rhs_verifier_mle,
            num_dataparallel_rounds: self.num_dataparallel_bits,
            claim_challenge_points: claim_point.to_vec(),
            dataparallel_sumcheck_challenges: dataparallel_challenges,
            first_u_challenges,
            last_v_challenges,
        };

        Ok(verifier_gate_layer)
    }

    fn get_post_sumcheck_layer(
        &self,
        _round_challenges: &[F],
        _claim_challenges: &[F],
    ) -> super::product::PostSumcheckLayer<F, Option<F>> {
        todo!()
    }

    fn max_degree(&self) -> usize {
        todo!()
    }

    fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        vec![&self.lhs_mle, &self.rhs_mle]
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let lhs_mle = self.lhs_mle.into_dense_mle(circuit_map);
        let rhs_mle = self.rhs_mle.into_dense_mle(circuit_map);
        let num_dataparallel_bits = if self.num_dataparallel_bits == 0 {
            None
        } else {
            Some(self.num_dataparallel_bits)
        };
        let gate_layer = GateLayer::new(
            num_dataparallel_bits,
            self.wiring.clone(),
            lhs_mle,
            rhs_mle,
            self.gate_operation,
            self.layer_id(),
        );
        gate_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.lhs_mle.index_mle_indices(start_index);
        self.rhs_mle.index_mle_indices(start_index);
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&MleDescription<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();

        let max_gate_val = self
            .wiring
            .iter()
            .fold(&0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (self.num_dataparallel_bits);
        let res_table_num_entries =
            ((max_gate_val + 1) * num_dataparallel_vals).next_power_of_two();

        let lhs_data = circuit_map
            .get_data_from_circuit_mle(&self.lhs_mle)
            .unwrap();
        let rhs_data = circuit_map
            .get_data_from_circuit_mle(&self.rhs_mle)
            .unwrap();

        let mut res_table = vec![F::ZERO; res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            self.wiring.iter().for_each(|(z_ind, x_ind, y_ind)| {
                let zero = F::ZERO;
                let f2_val = lhs_data
                    .get_evals_vector()
                    .get(idx + (x_ind * num_dataparallel_vals))
                    .unwrap_or(&zero);
                let f3_val = rhs_data
                    .get_evals_vector()
                    .get(idx + (y_ind * num_dataparallel_vals))
                    .unwrap_or(&zero);
                res_table[idx + (z_ind * num_dataparallel_vals)] =
                    self.gate_operation.perform_operation(*f2_val, *f3_val);
            });
        });

        let output_data = MultilinearExtension::new(res_table);
        assert_eq!(
            output_data.num_vars(),
            mle_output_necessary.var_indices().len()
        );

        circuit_map.add_node(CircuitLocation::new(self.layer_id(), vec![]), output_data);
    }
}

impl<F: Field> VerifierGateLayer<F> {
    /// Computes the oracle query's value for a given [VerifierGateLayer].
    pub fn evaluate(&self, claim: &Claim<F>) -> F {
        let g2_challenges = claim.get_point()[..self.num_dataparallel_rounds].to_vec();
        let g1_challenges = claim.get_point()[self.num_dataparallel_rounds..].to_vec();

        // Compute the gate function bound at those variables.
        // Beta table corresponding to the equality of binding the x variables to u.
        let beta_u = BetaValues::new_beta_equality_mle(self.first_u_challenges.clone());
        // Beta table corresponding to the equality of binding the y variables to v.
        let beta_v = BetaValues::new_beta_equality_mle(self.last_v_challenges.clone());
        // Beta table representing all "z" label challenges.
        let beta_g = BetaValues::new_beta_equality_mle(g1_challenges);
        // Multiply the corresponding entries of the beta tables to get the full value of the gate function
        // i.e. f1(z, x, y) bound at the challenges f1(g1, u, v).
        let f_1_uv = self
            .wiring
            .clone()
            .into_iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = *beta_u.bookkeeping_table().get(x_ind).unwrap_or(&F::ZERO);
                let vy = *beta_v.bookkeeping_table().get(y_ind).unwrap_or(&F::ZERO);
                acc + gz * ux * vy
            });

        // --- Finally, grab the claimed values for each of the bound MLEs from transcript ---
        // // First, the claimed value of V_{i + 1}(g_2, u)
        // let f2_bound = transcript_reader
        //     .consume_element("Evaluation of V_{i + 1}(g_2, u)")
        //     .unwrap();
        // // Next, the claimed value of V_{i + 1}(g_2, v)
        // let f3_bound = transcript_reader
        //     .consume_element("Evaluation of V_{i + 1}(g_2, v)")
        //     .unwrap();

        let beta_bound = BetaValues::compute_beta_over_two_challenges(
            &g2_challenges,
            &self.dataparallel_sumcheck_challenges,
        );

        // Compute the final result of the bound expression (this is the oracle query).
        beta_bound
            * (f_1_uv
                * self
                    .gate_operation
                    .perform_operation(self.lhs_mle.value(), self.rhs_mle.value()))
    }
}

/// The verifier's counterpart of a Gate layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct VerifierGateLayer<F: Field> {
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

    /// The challenge points for the claim on the [Gate] layer.
    claim_challenge_points: Vec<F>,

    /// The number of dataparallel rounds.
    num_dataparallel_rounds: usize,

    /// The challenges for `p_2`, as derived from sumcheck.
    dataparallel_sumcheck_challenges: Vec<F>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<F>,

    /// The challenges for `y`, as derived from sumcheck.
    last_v_challenges: Vec<F>,
}

impl<F: Field> VerifierLayer<F> for VerifierGateLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for GateLayer<F> {
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
            Some(self.lhs.layer_id()),
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
            Some(self.rhs.layer_id()),
        );
        claims.push(claim);

        Ok(claims)
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for VerifierGateLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        // Grab the claim on the left side.
        // TODO!(ryancao): Do error handling here!
        let lhs_vars = self.lhs_mle.mle_indices();
        let lhs_point = lhs_vars
            .iter()
            .map(|idx| match idx {
                MleIndex::Bound(chal, _bit_idx) => *chal,
                MleIndex::Fixed(val) => {
                    if *val {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                }
                _ => panic!("Error: Not fully bound"),
            })
            .collect_vec();
        let lhs_val = self.lhs_mle.value();

        let lhs_claim: ClaimMle<F> = ClaimMle::new(
            lhs_point,
            lhs_val,
            Some(self.layer_id()),
            Some(self.lhs_mle.layer_id()),
        );

        // Grab the claim on the right side.
        // TODO!(ryancao): Do error handling here!
        let rhs_vars: &[MleIndex<F>] = self.rhs_mle.mle_indices();
        let rhs_point = rhs_vars
            .iter()
            .map(|idx| match idx {
                MleIndex::Bound(chal, _bit_idx) => *chal,
                MleIndex::Fixed(val) => {
                    if *val {
                        F::ONE
                    } else {
                        F::ZERO
                    }
                }
                _ => panic!("Error: Not fully bound"),
            })
            .collect_vec();
        let rhs_val = self.rhs_mle.value();

        let rhs_claim: ClaimMle<F> = ClaimMle::new(
            rhs_point,
            rhs_val,
            Some(self.layer_id()),
            Some(self.rhs_mle.layer_id()),
        );

        Ok(vec![lhs_claim, rhs_claim])
    }
}

impl<F: Field> YieldWLXEvals<F> for GateLayer<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        _claimed_mles: Vec<DenseMle<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // Get the number of evaluations.
        let (num_evals, _, _) = get_num_wlx_evaluations(claim_vecs);

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

impl<F: Field> GateLayer<F> {
    /// Construct a new gate layer
    ///
    /// # Arguments
    /// * `num_dataparallel_bits`: an optional representing the number of bits representing the circuit copy we are looking at.
    ///
    /// None if this is not dataparallel, otherwise specify the number of bits
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    ///
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
        GateLayer {
            num_dataparallel_bits: num_dataparallel_bits.unwrap_or(0),
            nonzero_gates,
            lhs,
            rhs,
            layer_id,
            phase_1_mles: None,
            phase_2_mles: None,
            gate_operation,
            challenges: vec![],
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
    fn init_dataparallel_phase(
        &mut self,
        beta_g1: &mut DenseMle<F>,
        beta_g2: &mut DenseMle<F>,
    ) -> Result<Vec<F>, GateError> {
        // Index original bookkeeping tables.
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        // Result of initializing is the first sumcheck message.

        compute_sumcheck_messages_data_parallel_gate(
            &self.lhs,
            &self.rhs,
            beta_g2,
            beta_g1,
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
        let num_x = self.lhs.num_free_vars();

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

        // The actual mles differ based on whether we are doing a add gate or a mul gate, because
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

        let init_mle_refs: Vec<Vec<&DenseMle<F>>> = phase_1_mles
            .iter()
            .map(|mle_vec| {
                let mle_references: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                mle_references
            })
            .collect();
        let evals_vec = init_mle_refs
            .iter()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_bits, max_deg)
                    .unwrap()
            })
            .collect_vec();
        let final_evals = evals_vec
            .clone()
            .into_iter()
            .skip(1)
            .fold(SumcheckEvals(evals_vec[0].clone()), |acc, elem| {
                acc + SumcheckEvals(elem)
            });
        let SumcheckEvals(final_vec_evals) = final_evals;
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
        let num_y = self.rhs.num_free_vars();

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

        let init_mle_refs: Vec<Vec<&DenseMle<F>>> = phase_2_mles
            .iter()
            .map(|mle_vec| {
                let mle_references: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                mle_references
            })
            .collect();
        let evals_vec = init_mle_refs
            .iter()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_bits, max_deg)
                    .unwrap()
            })
            .collect_vec();
        let final_evals = evals_vec
            .clone()
            .into_iter()
            .skip(1)
            .fold(SumcheckEvals(evals_vec[0].clone()), |acc, elem| {
                acc + SumcheckEvals(elem)
            });
        let SumcheckEvals(final_vec_evals) = final_evals;
        Ok(final_vec_evals)
    }

    // Once the initialization of the dataparallel phase is done, we can perform the dataparallel phase.
    // This means that we are binding all bits that represent which copy of the circuit we are in.
    fn perform_dataparallel_phase(
        &mut self,
        beta_g1: &mut DenseMle<F>,
        beta_g2: &mut DenseMle<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(SumcheckProof<F>, F), LayerError> {
        // Initialization, first message comes from here.
        let mut challenges: Vec<F> = vec![];

        let first_message = self.init_dataparallel_phase(beta_g1, beta_g2).expect(
            "could not evaluate original lhs and rhs in order to get first sumcheck message",
        );

        let (lhs, rhs) = (&mut self.lhs, &mut self.rhs);

        transcript_writer
            .append_elements("Initial Sumcheck evaluations DATAPARALLEL", &first_message);
        let num_rounds_copy_phase = self.num_dataparallel_bits;

        // Do the first dataparallel bits number sumcheck rounds using libra giraffe.
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_copy_phase).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge DATAPARALLEL");
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
                transcript_writer.append_elements("Sumcheck evaluations DATAPARALLEL", &eval);
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // Bind the final challenge, update the final beta table.
        let final_chal_copy =
            transcript_writer.get_challenge("Final Sumcheck challenge DATAPARALLEL");
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
        transcript_writer: &mut impl ProverTranscript<F>,
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
        transcript_writer.append_elements("Sumcheck evaluations PHASE 1", &first_message);
        let num_rounds_phase1 = self.lhs.num_free_vars();

        // Sumcheck rounds (binding x).
        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_phase1).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge PHASE 1");
                challenges.push(challenge);
                // If there are dataparallel bits, we want to start at that index.
                bind_round_gate(round + self.num_dataparallel_bits, challenge, phase_1_mles);
                let phase_1_mle_refs: Vec<Vec<&DenseMle<F>>> = phase_1_mles
                    .iter()
                    .map(|mle_vec| {
                        let mle_references: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                        mle_references
                    })
                    .collect();
                let eval = compute_sumcheck_message_gate(
                    round + self.num_dataparallel_bits,
                    &phase_1_mle_refs,
                )
                .into_iter()
                .map(|eval| eval * beta_g2_fully_bound)
                .collect_vec();
                transcript_writer.append_elements("Sumcheck evaluations PHASE 1", &eval);
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
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let first_message = self
            .init_phase_2(phase_1_challenges, f_at_u, &beta_g1)
            .unwrap()
            .into_iter()
            .map(|eval| eval * beta_g2_fully_bound)
            .collect_vec();

        let mut challenges: Vec<F> = vec![];

        if self.rhs.num_free_vars() > 0 {
            let phase_2_mles = self
                .phase_2_mles
                .as_mut()
                .ok_or(GateError::Phase2InitError)
                .unwrap();

            transcript_writer.append_elements("Sumcheck evaluations", &first_message);

            let num_rounds_phase2 = self.rhs.num_free_vars();

            // Bind y, the right side of the sum.
            let sumcheck_rounds_y: Vec<Vec<F>> = std::iter::once(Ok(first_message))
                .chain((1..num_rounds_phase2).map(|round| {
                    let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                    challenges.push(challenge);
                    bind_round_gate(round + self.num_dataparallel_bits, challenge, phase_2_mles);
                    let phase_2_mle_refs: Vec<Vec<&DenseMle<F>>> = phase_2_mles
                        .iter()
                        .map(|mle_vec| {
                            let mle_references: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                            mle_references
                        })
                        .collect();
                    let eval = compute_sumcheck_message_gate(
                        round + self.num_dataparallel_bits,
                        &phase_2_mle_refs,
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
impl<F: std::fmt::Debug + Field> GateLayer<F> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct GateCircuitDesc<'a, F: std::fmt::Debug + Field>(&'a GateLayer<F>);

        impl<'a, F: std::fmt::Debug + Field> std::fmt::Display for GateCircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Gate")
                    .field("lhs_mle_ref_layer_id", &self.0.lhs.layer_id())
                    .field("lhs_mle_ref_mle_indices", &self.0.lhs.mle_indices())
                    .field("rhs_mle_ref_layer_id", &self.0.rhs.layer_id())
                    .field("rhs_mle_ref_mle_indices", &self.0.rhs.mle_indices())
                    .field("add_nonzero_gates", &self.0.nonzero_gates)
                    .field("num_dataparallel_bits", &self.0.num_dataparallel_bits)
                    .finish()
            }
        }
        GateCircuitDesc(self)
    }
}

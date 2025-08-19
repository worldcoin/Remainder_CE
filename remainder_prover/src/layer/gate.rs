//! module for defining the gate layer, uses the libra trick
//! to reduce the number of rounds for gate layers (with binary operations)

/// Helper functions used in the gate sumcheck algorithms.
pub mod gate_helpers;
use std::{cmp::max, collections::HashSet};

use gate_helpers::{
    compute_fully_bound_binary_gate_function, fold_binary_gate_wiring_into_mles_phase_1,
    fold_binary_gate_wiring_into_mles_phase_2,
};
use itertools::Itertools;
use remainder_shared_types::{
    config::{global_config::global_claim_agg_strategy, ClaimAggregationStrategy},
    extension_field::ExtensionField,
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    circuit_layout::{CircuitEvalMap, CircuitLocation},
    claims::{Claim, ClaimError, RawClaim},
    layer::{
        product::{PostSumcheckLayer, Product},
        Layer, LayerError, LayerId, VerificationError,
    },
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, verifier_mle::VerifierMle, AbstractMle, Mle, MleIndex,
    },
    sumcheck::{evaluate_at_a_point, SumcheckEvals},
};

use anyhow::{anyhow, Ok, Result};

pub use self::gate_helpers::{
    compute_sumcheck_message_data_parallel_gate, compute_sumcheck_message_no_beta_table,
    index_mle_indices_gate, GateError,
};

use super::{
    layer_enum::{LayerEnum, VerifierLayerEnum},
    LayerDescription, VerifierLayer,
};

#[derive(PartialEq, Serialize, Deserialize, Clone, Debug, Copy)]

/// Operations that are currently supported by the gate. Binary because these
/// are fan-in-two gates.
#[derive(Hash)]
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
/// is specified by `num_dataparallel_vars` in order to account for batched and un-batched
/// gates.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "E: ExtensionField")]
pub struct GateLayer<E: ExtensionField> {
    /// The layer id associated with this gate layer.
    pub layer_id: LayerId,
    /// The number of bits representing the number of "dataparallel" copies of the circuit.
    pub num_dataparallel_vars: usize,
    /// A vector of tuples representing the "nonzero" gates, especially useful in the sparse case
    /// the format is (z, x, y) where the gate at label z is the output of performing an operation
    /// on gates with labels x and y.
    pub nonzero_gates: Vec<(u32, u32, u32)>,
    /// The left side of the expression, i.e. the mle that makes up the "x" variables.
    pub lhs: DenseMle<E>,
    /// The right side of the expression, i.e. the mle that makes up the "y" variables.
    pub rhs: DenseMle<E>,
    /// The mles that are constructed when initializing phase 1 (binding the x variables).
    pub phase_1_mles: Option<Vec<Vec<DenseMle<E>>>>,
    /// The mles that are constructed when initializing phase 2 (binding the y variables).
    pub phase_2_mles: Option<Vec<Vec<DenseMle<E>>>>,
    /// The gate operation representing the fan-in-two relationship.
    pub gate_operation: BinaryOperation,
    /// the beta table which enumerates the incoming claim's challenge points on the
    /// dataparallel vars of the MLE
    beta_g2_vec: Option<Vec<BetaValues<E>>>,
    /// The incoming claim's challenge points.
    g_vec: Option<Vec<Vec<E>>>,
    /// the number of rounds in phase 1
    num_rounds_phase1: usize,
}

impl<E: ExtensionField> Layer<E> for GateLayer<E> {
    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn prove(
        &mut self,
        claims: &[&RawClaim<E>],
        transcript_writer: &mut impl ProverTranscript<E::BaseField>,
    ) -> Result<()> {
        let original_lhs_num_free_vars = self.lhs.num_free_vars();
        let original_rhs_num_free_vars = self.rhs.num_free_vars();
        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                assert_eq!(claims.len(), 1);
                self.initialize(claims[0].get_point())?;
                vec![E::ONE]
            }
            ClaimAggregationStrategy::RLC => {
                let random_coefficients = transcript_writer
                    .get_extension_field_challenges("RLC Claim Agg Coefficients", claims.len());
                self.initialize_rlc(&random_coefficients, claims);
                random_coefficients
            }
        };
        let sumcheck_indices = self.sumcheck_round_indices();
        (sumcheck_indices.iter()).for_each(|round_idx| {
            let sumcheck_message = self
                .compute_round_sumcheck_message(*round_idx, &random_coefficients)
                .unwrap();
            transcript_writer.append_extension_field_elements(
                "Sumcheck round univariate evaluations",
                &sumcheck_message,
            );
            let challenge =
                transcript_writer.get_extension_field_challenge("Sumcheck round challenge");
            self.bind_round_variable(*round_idx, challenge).unwrap();
        });

        // Edge case for if the LHS or RHS have 0 variables.
        if original_lhs_num_free_vars - self.num_dataparallel_vars == 0 {
            match global_claim_agg_strategy() {
                ClaimAggregationStrategy::Interpolative => {
                    self.init_phase_1(
                        self.g_vec.as_ref().unwrap()[0][self.num_dataparallel_vars..].to_vec(),
                    );
                }
                ClaimAggregationStrategy::RLC => {
                    self.init_phase_1_rlc(
                        &self
                            .g_vec
                            .as_ref()
                            .unwrap()
                            .clone()
                            .iter()
                            .map(|challenge| &challenge[self.num_dataparallel_vars..])
                            .collect_vec(),
                        &random_coefficients,
                    );
                }
            }
        }
        if original_rhs_num_free_vars - self.num_dataparallel_vars == 0 {
            let f2 = &self.phase_1_mles.as_ref().unwrap()[0][1];
            let f2_at_u = f2.value();
            let u_challenges = &f2
                .mle_indices()
                .iter()
                .filter_map(|mle_index| match mle_index {
                    MleIndex::Bound(value, _idx) => Some(*value),
                    MleIndex::Fixed(_) => None,
                    _ => panic!("Should not have any unbound values"),
                })
                .collect_vec()[self.num_dataparallel_vars..];

            match global_claim_agg_strategy() {
                ClaimAggregationStrategy::Interpolative => {
                    let g_challenges =
                        self.g_vec.as_ref().unwrap()[0][self.num_dataparallel_vars..].to_vec();
                    self.init_phase_2(u_challenges, f2_at_u, &g_challenges);
                }
                ClaimAggregationStrategy::RLC => {
                    self.init_phase_2_rlc(
                        u_challenges,
                        f2_at_u,
                        &self
                            .g_vec
                            .as_ref()
                            .unwrap()
                            .clone()
                            .iter()
                            .map(|claim| &claim[self.num_dataparallel_vars..])
                            .collect_vec(),
                        &random_coefficients,
                    );
                }
            }
        }

        // Finally, send the claimed values for each of the bound MLEs to the verifier
        // First, send the claimed value of V_{i + 1}(g_2, u)
        let lhs_reduced: &DenseMle<E> = &self.phase_1_mles.as_ref().unwrap()[0][1];
        let rhs_reduced: &DenseMle<E> = &self.phase_2_mles.as_ref().unwrap()[0][1];
        transcript_writer
            .append_extension_field_element("Fully bound MLE evaluation", lhs_reduced.value());
        // Next, send the claimed value of V_{i + 1}(g_2, v)
        transcript_writer
            .append_extension_field_element("Fully bound MLE evaluation", rhs_reduced.value());

        Ok(())
    }

    fn initialize(&mut self, claim_point: &[E]) -> Result<()> {
        self.beta_g2_vec = Some(vec![BetaValues::new(
            claim_point[..self.num_dataparallel_vars]
                .iter()
                .copied()
                .enumerate()
                .collect(),
        )]);
        self.g_vec = Some(vec![claim_point.to_vec()]);
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        Ok(())
    }

    fn initialize_rlc(&mut self, _random_coefficients: &[E], claims: &[&RawClaim<E>]) {
        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);
        let (g_vec, beta_g2_vec): (Vec<Vec<E>>, Vec<BetaValues<E>>) = claims
            .iter()
            .map(|claim| {
                (
                    claim.get_point().to_vec(),
                    BetaValues::new(
                        claim.get_point()[..self.num_dataparallel_vars]
                            .iter()
                            .copied()
                            .enumerate()
                            .collect(),
                    ),
                )
            })
            .unzip();
        self.g_vec = Some(g_vec);
        self.beta_g2_vec = Some(beta_g2_vec);
    }

    fn compute_round_sumcheck_message(
        &mut self,
        round_index: usize,
        random_coefficients: &[E],
    ) -> Result<Vec<E>> {
        let rounds_before_phase_2 = self.num_dataparallel_vars + self.num_rounds_phase1;

        if round_index < self.num_dataparallel_vars {
            // dataparallel phase
            Ok(compute_sumcheck_message_data_parallel_gate(
                &self.lhs,
                &self.rhs,
                self.gate_operation,
                &self.nonzero_gates,
                self.num_dataparallel_vars - round_index,
                &self
                    .g_vec
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|challenge| &challenge[round_index..])
                    .collect_vec(),
                &self
                    .beta_g2_vec
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(random_coefficients)
                    .map(|(beta_values, random_coeff)| {
                        *random_coeff * beta_values.fold_updated_values()
                    })
                    .collect_vec(),
            )
            .unwrap())
        } else if round_index < rounds_before_phase_2 {
            if round_index == self.num_dataparallel_vars {
                match global_claim_agg_strategy() {
                    ClaimAggregationStrategy::Interpolative => {
                        self.init_phase_1(
                            self.g_vec.as_ref().unwrap()[0][self.num_dataparallel_vars..].to_vec(),
                        );
                    }
                    ClaimAggregationStrategy::RLC => {
                        self.init_phase_1_rlc(
                            &self
                                .g_vec
                                .as_ref()
                                .unwrap()
                                .clone()
                                .iter()
                                .map(|challenge| &challenge[self.num_dataparallel_vars..])
                                .collect_vec(),
                            random_coefficients,
                        );
                    }
                }
            }
            let max_deg = self
                .phase_1_mles
                .as_ref()
                .unwrap()
                .iter()
                .fold(0, |acc, elem| max(acc, elem.len()));

            let init_mles: Vec<Vec<&DenseMle<E>>> = self
                .phase_1_mles
                .as_ref()
                .unwrap()
                .iter()
                .map(|mle_vec| {
                    let mle_reference = mle_vec.iter().collect();
                    mle_reference
                })
                .collect();
            let evals_vec = init_mles
                .iter()
                .map(|mle_vec| {
                    compute_sumcheck_message_no_beta_table(mle_vec, round_index, max_deg).unwrap()
                })
                .collect_vec();
            let final_evals = evals_vec
                .clone()
                .into_iter()
                .skip(1)
                .fold(SumcheckEvals(evals_vec[0].clone()), |acc, elem| {
                    acc + SumcheckEvals(elem)
                });

            Ok(final_evals.0)
        } else {
            if round_index == rounds_before_phase_2 {
                let f2 = &self.phase_1_mles.as_ref().unwrap()[0][1];
                let f2_at_u = f2.value();
                let u_challenges = &f2
                    .mle_indices()
                    .iter()
                    .filter_map(|mle_index| match mle_index {
                        MleIndex::Bound(value, _idx) => Some(*value),
                        MleIndex::Fixed(_) => None,
                        _ => panic!("Should not have any unbound values"),
                    })
                    .collect_vec()[self.num_dataparallel_vars..];

                match global_claim_agg_strategy() {
                    ClaimAggregationStrategy::Interpolative => {
                        let g_challenges =
                            self.g_vec.as_ref().unwrap()[0][self.num_dataparallel_vars..].to_vec();
                        self.init_phase_2(u_challenges, f2_at_u, &g_challenges);
                    }
                    ClaimAggregationStrategy::RLC => {
                        self.init_phase_2_rlc(
                            u_challenges,
                            f2_at_u,
                            &self
                                .g_vec
                                .as_ref()
                                .unwrap()
                                .clone()
                                .iter()
                                .map(|claim| &claim[self.num_dataparallel_vars..])
                                .collect_vec(),
                            random_coefficients,
                        );
                    }
                }
            }
            if self.phase_2_mles.as_ref().unwrap()[0][1].num_free_vars() > 0 {
                // Return the first sumcheck message of this phase.
                let max_deg = self
                    .phase_2_mles
                    .as_ref()
                    .unwrap()
                    .iter()
                    .fold(0, |acc, elem| max(acc, elem.len()));

                let init_mles: Vec<Vec<&DenseMle<E>>> = self
                    .phase_2_mles
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|mle_vec| {
                        let mle_references: Vec<&DenseMle<E>> = mle_vec.iter().collect();
                        mle_references
                    })
                    .collect();
                let evals_vec = init_mles
                    .iter()
                    .map(|mle_vec| {
                        compute_sumcheck_message_no_beta_table(
                            mle_vec,
                            round_index - self.num_rounds_phase1,
                            max_deg,
                        )
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
                Ok(final_evals.0)
            } else {
                Ok(vec![])
            }
        }
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: E) -> Result<()> {
        if round_index < self.num_dataparallel_vars {
            self.beta_g2_vec
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|beta| beta.beta_update(round_index, challenge));
            self.lhs.fix_variable(round_index, challenge);
            self.rhs.fix_variable(round_index, challenge);

            Ok(())
        } else if round_index < self.num_rounds_phase1 + self.num_dataparallel_vars {
            let mles = self.phase_1_mles.as_mut().unwrap();
            mles.iter_mut().for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.fix_variable(round_index, challenge);
                })
            });
            Ok(())
        } else {
            let round_index = round_index - self.num_rounds_phase1;
            let mles = self.phase_2_mles.as_mut().unwrap();
            mles.iter_mut().for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.fix_variable(round_index, challenge);
                })
            });
            Ok(())
        }
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        let num_u = self.lhs.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;

        (0..num_u + num_v + self.num_dataparallel_vars).collect_vec()
    }

    fn max_degree(&self) -> usize {
        match self.gate_operation {
            BinaryOperation::Add => 2,
            BinaryOperation::Mul => {
                if self.num_dataparallel_vars != 0 {
                    3
                } else {
                    2
                }
            }
        }
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[E],
        claim_challenges: &[&[E]],
        random_coefficients: &[E],
    ) -> super::product::PostSumcheckLayer<E, E> {
        assert_eq!(claim_challenges.len(), random_coefficients.len());
        let lhs_mle = &self.phase_1_mles.as_ref().unwrap()[0][1];
        let rhs_mle = &self.phase_2_mles.as_ref().unwrap()[0][1];

        let g2_challenges_vec = claim_challenges
            .iter()
            .map(|claim_chal| &claim_chal[..self.num_dataparallel_vars])
            .collect_vec();
        let g1_challenges_vec = claim_challenges
            .iter()
            .map(|claim_chal| &claim_chal[self.num_dataparallel_vars..])
            .collect_vec();

        let dataparallel_sumcheck_challenges =
            round_challenges[..self.num_dataparallel_vars].to_vec();
        let first_u_challenges = round_challenges
            [self.num_dataparallel_vars..self.num_dataparallel_vars + self.num_rounds_phase1]
            .to_vec();
        let last_v_challenges =
            round_challenges[self.num_dataparallel_vars + self.num_rounds_phase1..].to_vec();
        let random_coefficients_scaled_by_beta_bound = g2_challenges_vec
            .iter()
            .zip(random_coefficients)
            .map(|(g2_challenges, random_coeff)| {
                let beta_bound = if self.num_dataparallel_vars != 0 {
                    BetaValues::compute_beta_over_two_challenges(
                        g2_challenges,
                        &dataparallel_sumcheck_challenges,
                    )
                } else {
                    E::ONE
                };
                beta_bound * random_coeff
            })
            .collect_vec();

        let f_1_uv = compute_fully_bound_binary_gate_function(
            &first_u_challenges,
            &last_v_challenges,
            &g1_challenges_vec,
            &self.nonzero_gates,
            &random_coefficients_scaled_by_beta_bound,
        );

        match self.gate_operation {
            BinaryOperation::Add => PostSumcheckLayer(vec![
                Product::<E, E>::new(&[lhs_mle.clone()], f_1_uv),
                Product::<E, E>::new(&[rhs_mle.clone()], f_1_uv),
            ]),
            BinaryOperation::Mul => PostSumcheckLayer(vec![Product::<E, E>::new(
                &[lhs_mle.clone(), rhs_mle.clone()],
                f_1_uv,
            )]),
        }
    }

    fn get_claims(&self) -> Result<Vec<Claim<E>>> {
        let lhs_reduced = self.phase_1_mles.clone().unwrap()[0][1].clone();
        let rhs_reduced = self.phase_2_mles.clone().unwrap()[0][1].clone();

        let mut claims = vec![];

        // Grab the claim on the left side.
        let mut fixed_mle_indices_u: Vec<E> = vec![];
        for index in lhs_reduced.mle_indices() {
            fixed_mle_indices_u.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = lhs_reduced.value();
        let claim: Claim<E> = Claim::new(
            fixed_mle_indices_u,
            val,
            self.layer_id(),
            self.lhs.layer_id(),
        );
        claims.push(claim);

        // Grab the claim on the right side.
        let mut fixed_mle_indices_v: Vec<E> = vec![];
        for index in rhs_reduced.mle_indices() {
            fixed_mle_indices_v.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = rhs_reduced.value();
        let claim: Claim<E> = Claim::new(
            fixed_mle_indices_v,
            val,
            self.layer_id(),
            self.rhs.layer_id(),
        );
        claims.push(claim);

        Ok(claims)
    }
}

/// The circuit-description counterpart of a Gate layer description.
#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
#[serde(bound = "E: ExtensionField")]
pub struct GateLayerDescription<E: ExtensionField> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// The gate operation representing the fan-in-two relationship.
    gate_operation: BinaryOperation,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x, y) where the gate at label z is
    /// the output of performing an operation on gates with labels x and y.
    nonzero_gates: Vec<(u32, u32, u32)>,

    /// The left side of the expression, i.e. the mle that makes up the "x"
    /// variables.
    lhs_mle: MleDescription<E>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: MleDescription<E>,

    /// The number of bits representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_vars: usize,
}

impl<E: ExtensionField> GateLayerDescription<E> {
    /// Constructor for a [GateLayerDescription].
    pub fn new(
        num_dataparallel_vars: Option<usize>,
        wiring: Vec<(u32, u32, u32)>,
        lhs_circuit_mle: MleDescription<E>,
        rhs_circuit_mle: MleDescription<E>,
        gate_layer_id: LayerId,
        gate_operation: BinaryOperation,
    ) -> Self {
        GateLayerDescription {
            id: gate_layer_id,
            gate_operation,
            nonzero_gates: wiring,
            lhs_mle: lhs_circuit_mle,
            rhs_mle: rhs_circuit_mle,
            num_dataparallel_vars: num_dataparallel_vars.unwrap_or(0),
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

impl<E: ExtensionField> LayerDescription<E> for GateLayerDescription<E> {
    type VerifierLayer = VerifierGateLayer<E>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn verify_rounds(
        &self,
        claims: &[&RawClaim<E>],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<VerifierLayerEnum<E>> {
        // Storing challenges for the sake of claim generation later
        let mut challenges = vec![];

        // Random coefficients depending on claim aggregation strategy.
        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                assert_eq!(claims.len(), 1);
                vec![E::ONE]
            }
            ClaimAggregationStrategy::RLC => transcript_reader
                .get_extension_field_challenges("RLC Claim Agg Coefficients", claims.len())?,
        };

        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED
        // BITS
        let num_u = self.lhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;

        // Store all prover sumcheck messages to check against
        let mut sumcheck_messages: Vec<Vec<E>> = vec![];

        // First round check against the claim.
        let first_round_num_evals = match (self.gate_operation, self.num_dataparallel_vars) {
            (BinaryOperation::Add, 0) => NON_DATAPARALLEL_ROUND_ADD_NUM_EVALS,
            (BinaryOperation::Mul, 0) => NON_DATAPARALLEL_ROUND_MUL_NUM_EVALS,
            (BinaryOperation::Add, _) => DATAPARALLEL_ROUND_ADD_NUM_EVALS,
            (BinaryOperation::Mul, _) => DATAPARALLEL_ROUND_MUL_NUM_EVALS,
        };
        let first_round_sumcheck_messages = transcript_reader.consume_extension_field_elements(
            "Sumcheck round univariate evaluations",
            first_round_num_evals,
        )?;
        sumcheck_messages.push(first_round_sumcheck_messages.clone());

        match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                // Check: V_i(g_2, g_1) =? g_1(0) + g_1(1)
                // TODO(ryancao): SUPER overloaded notation (in e.g. above comments); fix across the board
                if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1]
                    != claims[0].get_eval()
                {
                    return Err(anyhow!(VerificationError::SumcheckStartFailed));
                }
            }
            ClaimAggregationStrategy::RLC => {
                let rlc_claim_eval = random_coefficients
                    .iter()
                    .zip(claims)
                    .fold(E::ZERO, |acc, (rlc_val, claim)| {
                        acc + *rlc_val * claim.get_eval()
                    });
                if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1]
                    != rlc_claim_eval
                {
                    return Err(anyhow!(VerificationError::SumcheckStartFailed));
                }
            }
        }

        // Check each of the messages -- note that here the verifier doesn't actually see the difference
        // between dataparallel rounds, phase 1 rounds, and phase 2 rounds; instead, the prover's proof reads
        // as a single continuous proof.
        for sumcheck_round_idx in 1..self.num_dataparallel_vars + num_u + num_v {
            // Read challenge r_{i - 1} from transcript
            let challenge = transcript_reader
                .get_extension_field_challenge("Sumcheck round challenge")
                .unwrap();
            let g_i_minus_1_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();

            // Evaluate g_{i - 1}(r_{i - 1})
            let prev_at_r = evaluate_at_a_point(&g_i_minus_1_evals, challenge).unwrap();

            // Read off g_i(0), g_i(1), ..., g_i(d) from transcript
            let univariate_num_evals = match (
                sumcheck_round_idx < self.num_dataparallel_vars, // 0-indexed, so strictly less-than is correct
                self.gate_operation,
            ) {
                (true, BinaryOperation::Add) => DATAPARALLEL_ROUND_ADD_NUM_EVALS,
                (true, BinaryOperation::Mul) => DATAPARALLEL_ROUND_MUL_NUM_EVALS,
                (false, BinaryOperation::Add) => NON_DATAPARALLEL_ROUND_ADD_NUM_EVALS,
                (false, BinaryOperation::Mul) => NON_DATAPARALLEL_ROUND_MUL_NUM_EVALS,
            };

            let curr_evals = transcript_reader
                .consume_extension_field_elements(
                    "Sumcheck round univariate evaluations",
                    univariate_num_evals,
                )
                .unwrap();

            // Check: g_i(0) + g_i(1) =? g_{i - 1}(r_{i - 1})
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                dbg!(&sumcheck_round_idx);
                return Err(anyhow!(VerificationError::SumcheckFailed));
            };

            // Add the prover message to the sumcheck messages
            sumcheck_messages.push(curr_evals);
            // Add the challenge.
            challenges.push(challenge);
        }

        // Final round of sumcheck -- sample r_n from transcript.
        let final_chal = transcript_reader
            .get_extension_field_challenge("Sumcheck round challenge")
            .unwrap();
        challenges.push(final_chal);

        // Create the resulting verifier layer for claim tracking
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_gate_layer = self
            .convert_into_verifier_layer(
                &challenges,
                &claims.iter().map(|claim| claim.get_point()).collect_vec(),
                transcript_reader,
            )
            .unwrap();
        let final_result = verifier_gate_layer.evaluate(
            &claims.iter().map(|claim| claim.get_point()).collect_vec(),
            &random_coefficients,
        );

        // Finally, compute g_n(r_n).
        let g_n_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();
        let prev_at_r = evaluate_at_a_point(&g_n_evals, final_chal).unwrap();

        // Final check in sumcheck.
        if final_result != prev_at_r {
            return Err(anyhow!(VerificationError::FinalSumcheckFailed));
        }

        Ok(VerifierLayerEnum::Gate(verifier_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        let num_u = self.lhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        (0..num_u + num_v + self.num_dataparallel_vars).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[E],
        claim_points: &[&[E]],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<Self::VerifierLayer> {
        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED
        // BITS
        let num_u = self.lhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.mle_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;

        // We want to separate the challenges into which ones are from the dataparallel bits, which ones
        // are for binding x (phase 1), and which are for binding y (phase 2).
        let mut sumcheck_bindings_vec = sumcheck_bindings.to_vec();
        let last_v_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_vars + num_u);
        let first_u_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_vars);
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

        // Create the resulting verifier layer for claim tracking
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_gate_layer = VerifierGateLayer {
            layer_id: self.layer_id(),
            gate_operation: self.gate_operation,
            wiring: self.nonzero_gates.clone(),
            lhs_mle: lhs_verifier_mle,
            rhs_mle: rhs_verifier_mle,
            num_dataparallel_rounds: self.num_dataparallel_vars,
            claim_challenge_points: claim_points
                .iter()
                .cloned()
                .map(|claim| claim.to_vec())
                .collect_vec(),
            dataparallel_sumcheck_challenges: dataparallel_challenges,
            first_u_challenges,
            last_v_challenges,
        };

        Ok(verifier_gate_layer)
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[E],
        claim_challenges: &[&[E]],
        random_coefficients: &[E],
    ) -> super::product::PostSumcheckLayer<E, Option<E>> {
        let num_rounds_phase1 = self.lhs_mle.num_free_vars() - self.num_dataparallel_vars;

        let g2_challenges_vec = claim_challenges
            .iter()
            .map(|claim_chal| &claim_chal[..self.num_dataparallel_vars])
            .collect_vec();
        let g1_challenges_vec = claim_challenges
            .iter()
            .map(|claim_chal| &claim_chal[self.num_dataparallel_vars..])
            .collect_vec();

        let dataparallel_sumcheck_challenges =
            round_challenges[..self.num_dataparallel_vars].to_vec();
        let first_u_challenges = round_challenges
            [self.num_dataparallel_vars..self.num_dataparallel_vars + num_rounds_phase1]
            .to_vec();
        let last_v_challenges =
            round_challenges[self.num_dataparallel_vars + num_rounds_phase1..].to_vec();
        let random_coefficients_scaled_by_beta_bound = g2_challenges_vec
            .iter()
            .zip(random_coefficients)
            .map(|(g2_challenges, random_coeff)| {
                let beta_bound = if self.num_dataparallel_vars != 0 {
                    BetaValues::compute_beta_over_two_challenges(
                        g2_challenges,
                        &dataparallel_sumcheck_challenges,
                    )
                } else {
                    E::ONE
                };
                beta_bound * random_coeff
            })
            .collect_vec();

        let f_1_uv = compute_fully_bound_binary_gate_function(
            &first_u_challenges,
            &last_v_challenges,
            &g1_challenges_vec,
            &self.nonzero_gates,
            &random_coefficients_scaled_by_beta_bound,
        );
        let lhs_challenges = &round_challenges[..self.num_dataparallel_vars + num_rounds_phase1];
        let rhs_challenges = &round_challenges[..self.num_dataparallel_vars]
            .iter()
            .copied()
            .chain(round_challenges[self.num_dataparallel_vars + num_rounds_phase1..].to_vec())
            .collect_vec();

        match self.gate_operation {
            BinaryOperation::Add => PostSumcheckLayer(vec![
                Product::<E, Option<E>>::new(&[self.lhs_mle.clone()], f_1_uv, lhs_challenges),
                Product::<E, Option<E>>::new(&[self.rhs_mle.clone()], f_1_uv, rhs_challenges),
            ]),
            BinaryOperation::Mul => {
                PostSumcheckLayer(vec![Product::<E, Option<E>>::new_from_mul_gate(
                    &[self.lhs_mle.clone(), self.rhs_mle.clone()],
                    f_1_uv,
                    &[lhs_challenges, rhs_challenges],
                )])
            }
        }
    }

    fn max_degree(&self) -> usize {
        match self.gate_operation {
            BinaryOperation::Add => 2,
            BinaryOperation::Mul => {
                if self.num_dataparallel_vars != 0 {
                    3
                } else {
                    2
                }
            }
        }
    }

    fn get_circuit_mles(&self) -> Vec<&MleDescription<E>> {
        vec![&self.lhs_mle, &self.rhs_mle]
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitEvalMap<E>) -> LayerEnum<E> {
        let lhs_mle = self.lhs_mle.into_dense_mle(circuit_map);
        let rhs_mle = self.rhs_mle.into_dense_mle(circuit_map);
        let num_dataparallel_vars = if self.num_dataparallel_vars == 0 {
            None
        } else {
            Some(self.num_dataparallel_vars)
        };
        let gate_layer = GateLayer::new(
            num_dataparallel_vars,
            self.nonzero_gates.clone(),
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
        mle_outputs_necessary: &HashSet<&MleDescription<E>>,
        circuit_map: &mut CircuitEvalMap<E>,
    ) {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();

        let max_gate_val = self
            .nonzero_gates
            .iter()
            .fold(&0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (self.num_dataparallel_vars);
        let res_table_num_entries =
            ((max_gate_val + 1) * num_dataparallel_vals).next_power_of_two();

        let lhs_data = circuit_map
            .get_data_from_circuit_mle(&self.lhs_mle)
            .unwrap();
        let rhs_data = circuit_map
            .get_data_from_circuit_mle(&self.rhs_mle)
            .unwrap();

        let num_gate_outputs_per_dataparallel_instance = (max_gate_val + 1).next_power_of_two();
        let mut res_table = vec![E::ZERO; res_table_num_entries as usize];
        (0..num_dataparallel_vals).for_each(|idx| {
            self.nonzero_gates.iter().for_each(|(z_ind, x_ind, y_ind)| {
                let zero = E::ZERO;
                let f2_val = lhs_data
                    .f
                    .get(
                        (idx * (1 << (lhs_data.num_vars() - self.num_dataparallel_vars)) + x_ind)
                            as usize,
                    )
                    .unwrap_or(zero);
                let f3_val = rhs_data
                    .f
                    .get(
                        (idx * (1 << (rhs_data.num_vars() - self.num_dataparallel_vars)) + y_ind)
                            as usize,
                    )
                    .unwrap_or(zero);
                res_table[(num_gate_outputs_per_dataparallel_instance * idx + z_ind) as usize] +=
                    self.gate_operation.perform_operation(f2_val, f3_val);
            });
        });

        let output_data = MultilinearExtension::new(res_table);
        assert_eq!(
            output_data.num_vars(),
            mle_output_necessary.mle_indices().len()
        );

        circuit_map.add_node(CircuitLocation::new(self.layer_id(), vec![]), output_data);
    }
}

impl<E: ExtensionField> VerifierGateLayer<E> {
    /// Computes the oracle query's value for a given [VerifierGateLayer].
    pub fn evaluate(&self, claims: &[&[E]], random_coefficients: &[E]) -> E {
        assert_eq!(random_coefficients.len(), claims.len());
        let scaled_random_coeffs = claims
            .iter()
            .zip(random_coefficients)
            .map(|(claim, random_coeff)| {
                let beta_bound = BetaValues::compute_beta_over_two_challenges(
                    &claim[..self.num_dataparallel_rounds],
                    &self.dataparallel_sumcheck_challenges,
                );
                beta_bound * random_coeff
            })
            .collect_vec();

        let f_1_uv = compute_fully_bound_binary_gate_function(
            &self.first_u_challenges,
            &self.last_v_challenges,
            &claims
                .iter()
                .map(|claim| &claim[self.num_dataparallel_rounds..])
                .collect_vec(),
            &self.wiring,
            &scaled_random_coeffs,
        );

        // Compute the final result of the bound expression (this is the oracle query).
        f_1_uv
            * self
                .gate_operation
                .perform_operation(self.lhs_mle.value(), self.rhs_mle.value())
    }
}

/// The verifier's counterpart of a Gate layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "E: ExtensionField")]
pub struct VerifierGateLayer<E: ExtensionField> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// The gate operation representing the fan-in-two relationship.
    gate_operation: BinaryOperation,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x, y) where the gate at label z is
    /// the output of performing an operation on gates with labels x and y.
    wiring: Vec<(u32, u32, u32)>,

    /// The left side of the expression, i.e. the mle that makes up the "x"
    /// variables.
    lhs_mle: VerifierMle<E>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: VerifierMle<E>,

    /// The challenge points for the claim on the [VerifierGateLayer].
    claim_challenge_points: Vec<Vec<E>>,

    /// The number of dataparallel rounds.
    num_dataparallel_rounds: usize,

    /// The challenges for `p_2`, as derived from sumcheck.
    dataparallel_sumcheck_challenges: Vec<E>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<E>,

    /// The challenges for `y`, as derived from sumcheck.
    last_v_challenges: Vec<E>,
}

impl<E: ExtensionField> VerifierLayer<E> for VerifierGateLayer<E> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_claims(&self) -> Result<Vec<Claim<E>>> {
        // Grab the claim on the left side.
        // TODO!(ryancao): Do error handling here!
        let lhs_vars = self.lhs_mle.mle_indices();
        let lhs_point = lhs_vars
            .iter()
            .map(|idx| match idx {
                MleIndex::Bound(chal, _bit_idx) => *chal,
                MleIndex::Fixed(val) => {
                    if *val {
                        E::ONE
                    } else {
                        E::ZERO
                    }
                }
                _ => panic!("Error: Not fully bound"),
            })
            .collect_vec();
        let lhs_val = self.lhs_mle.value();

        let lhs_claim: Claim<E> =
            Claim::new(lhs_point, lhs_val, self.layer_id(), self.lhs_mle.layer_id());

        // Grab the claim on the right side.
        // TODO!(ryancao): Do error handling here!
        let rhs_vars: &[MleIndex<E>] = self.rhs_mle.mle_indices();
        let rhs_point = rhs_vars
            .iter()
            .map(|idx| match idx {
                MleIndex::Bound(chal, _bit_idx) => *chal,
                MleIndex::Fixed(val) => {
                    if *val {
                        E::ONE
                    } else {
                        E::ZERO
                    }
                }
                _ => panic!("Error: Not fully bound"),
            })
            .collect_vec();
        let rhs_val = self.rhs_mle.value();

        let rhs_claim: Claim<E> =
            Claim::new(rhs_point, rhs_val, self.layer_id(), self.rhs_mle.layer_id());

        Ok(vec![lhs_claim, rhs_claim])
    }
}

impl<E: ExtensionField> GateLayer<E> {
    /// Construct a new gate layer
    ///
    /// # Arguments
    /// * `num_dataparallel_vars`: an optional representing the number of bits representing the circuit copy we are looking at.
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
        num_dataparallel_vars: Option<usize>,
        nonzero_gates: Vec<(u32, u32, u32)>,
        lhs: DenseMle<E>,
        rhs: DenseMle<E>,
        gate_operation: BinaryOperation,
        layer_id: LayerId,
    ) -> Self {
        let num_dataparallel_vars = num_dataparallel_vars.unwrap_or(0);
        let num_rounds_phase1 = lhs.num_free_vars() - num_dataparallel_vars;

        GateLayer {
            num_dataparallel_vars,
            nonzero_gates,
            lhs: lhs,
            rhs: rhs,
            layer_id,
            phase_1_mles: None,
            phase_2_mles: None,
            gate_operation,
            beta_g2_vec: None,
            g_vec: None,
            num_rounds_phase1,
        }
    }

    /// Initialize phase 1, or the necessary mles in order to bind the variables in the `lhs` of the
    /// expression. Once this phase is initialized, the sumcheck rounds binding the "x" variables can
    /// be performed.
    fn init_phase_1(&mut self, challenges: Vec<E>) {
        let beta_g2_fully_bound = self.beta_g2_vec.as_ref().unwrap()[0].fold_updated_values();

        let (a_hg_lhs_vec, a_hg_rhs_vec) = fold_binary_gate_wiring_into_mles_phase_1(
            &self.nonzero_gates,
            &[&challenges],
            &self.lhs,
            &self.rhs,
            &[beta_g2_fully_bound],
            self.gate_operation,
        );

        // The actual mles differ based on whether we are doing a add gate or a mul gate, because
        // in the case of an add gate, we distribute the gate function whereas in the case of the
        // mul gate, we simply take the product over all three mles.
        let mut phase_1_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_hg_lhs_vec, LayerId::Input(0)),
                        self.lhs.clone(),
                    ],
                    vec![DenseMle::new_from_raw(a_hg_rhs_vec, LayerId::Input(0))],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![
                    DenseMle::new_from_raw(a_hg_rhs_vec, LayerId::Input(0)),
                    self.lhs.clone(),
                ]]
            }
        };

        phase_1_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });
        self.phase_1_mles = Some(phase_1_mles);
    }

    fn init_phase_1_rlc(&mut self, challenges: &[&[E]], random_coefficients: &[E]) {
        let random_coefficients_scaled_by_beta_g2 = self
            .beta_g2_vec
            .as_ref()
            .unwrap()
            .iter()
            .zip(random_coefficients)
            .map(|(beta_values, random_coeff)| {
                assert!(beta_values.is_fully_bounded());
                beta_values.fold_updated_values() * random_coeff
            })
            .collect_vec();

        let (a_hg_lhs_vec, a_hg_rhs_vec) = fold_binary_gate_wiring_into_mles_phase_1(
            &self.nonzero_gates,
            challenges,
            &self.lhs,
            &self.rhs,
            &random_coefficients_scaled_by_beta_g2,
            self.gate_operation,
        );

        // The actual mles differ based on whether we are doing a add gate or a mul gate, because
        // in the case of an add gate, we distribute the gate function whereas in the case of the
        // mul gate, we simply take the product over all three mles.
        let mut phase_1_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_hg_lhs_vec, LayerId::Input(0)),
                        self.lhs.clone(),
                    ],
                    vec![DenseMle::new_from_raw(a_hg_rhs_vec, LayerId::Input(0))],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![
                    DenseMle::new_from_raw(a_hg_rhs_vec, LayerId::Input(0)),
                    self.lhs.clone(),
                ]]
            }
        };

        phase_1_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });
        self.phase_1_mles = Some(phase_1_mles);
    }

    /// Initialize phase 2, or the necessary mles in order to bind the variables in the `rhs` of the
    /// expression. Once this phase is initialized, the sumcheck rounds binding the "y" variables can
    /// be performed.
    fn init_phase_2(&mut self, u_claim: &[E], f_at_u: E, g1_claim_points: &[E]) {
        let beta_g2_fully_bound = self.beta_g2_vec.as_ref().unwrap()[0].fold_updated_values();

        let (a_f1_lhs, a_f1_rhs) = fold_binary_gate_wiring_into_mles_phase_2(
            &self.nonzero_gates,
            f_at_u,
            u_claim,
            &[g1_claim_points],
            &[beta_g2_fully_bound],
            self.rhs.num_free_vars(),
            self.gate_operation,
        );

        // We need to multiply h_g(x) by f_2(x)
        let mut phase_2_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0)),
                        self.rhs.clone(),
                    ],
                    vec![DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0))],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![
                    DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0)),
                    self.rhs.clone(),
                ]]
            }
        };

        phase_2_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });
        self.phase_2_mles = Some(phase_2_mles);
    }

    fn init_phase_2_rlc(
        &mut self,
        u_claim: &[E],
        f_at_u: E,
        g1_claim_points: &[&[E]],
        random_coefficients: &[E],
    ) {
        let random_coefficients_scaled_by_beta_g2 = self
            .beta_g2_vec
            .as_ref()
            .unwrap()
            .iter()
            .zip(random_coefficients)
            .map(|(beta_values, random_coeff)| {
                assert!(beta_values.is_fully_bounded());
                beta_values.fold_updated_values() * random_coeff
            })
            .collect_vec();

        let (a_f1_lhs, a_f1_rhs) = fold_binary_gate_wiring_into_mles_phase_2(
            &self.nonzero_gates,
            f_at_u,
            u_claim,
            g1_claim_points,
            &random_coefficients_scaled_by_beta_g2,
            self.rhs.num_free_vars(),
            self.gate_operation,
        );

        // We need to multiply h_g(x) by f_2(x)
        let mut phase_2_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0)),
                        self.rhs.clone(),
                    ],
                    vec![DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0))],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![
                    DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0)),
                    self.rhs.clone(),
                ]]
            }
        };

        phase_2_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });
        self.phase_2_mles = Some(phase_2_mles.clone());
    }
}

/// Computes the correct result of a gate layer,
/// Used for data generation and testing.
/// Arguments:
/// - wiring: A vector of tuples representing the "nonzero" gates, especially useful
///   in the sparse case the format is (z, x, y) where the gate at label z is
///   the output of performing an operation on gates with labels x and y.
///
/// - num_dataparallel_bits: The number of bits representing the number of "dataparallel"
///   copies of the circuit.
///
/// - lhs_data: The left side of the expression, i.e. the mle that makes up the "x"
///   variables.
///
/// - rhs_data: The mles that are constructed when initializing phase 2 (binding the y
///   variables).
///
/// - gate_operation: The gate operation representing the fan-in-two relationship.
pub fn compute_gate_data_outputs<F: Field>(
    wiring: Vec<(u32, u32, u32)>,
    num_dataparallel_bits: usize,
    lhs_data: &MultilinearExtension<F>,
    rhs_data: &MultilinearExtension<F>,
    gate_operation: BinaryOperation,
) -> MultilinearExtension<F> {
    let max_gate_val = wiring
        .iter()
        .fold(&0, |acc, (z, _, _)| std::cmp::max(acc, z));

    // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
    // evaluating over all values in the boolean hypercube which includes dataparallel bits
    let num_dataparallel_vals = 1 << num_dataparallel_bits;
    let res_table_num_entries = ((max_gate_val + 1) * num_dataparallel_vals).next_power_of_two();
    let num_gate_outputs_per_dataparallel_instance = (max_gate_val + 1).next_power_of_two();

    let mut res_table = vec![F::ZERO; res_table_num_entries as usize];
    // TDH(ende): investigate if this can be parallelized (and if it's a bottleneck)
    (0..num_dataparallel_vals).for_each(|idx| {
        wiring.iter().for_each(|(z_ind, x_ind, y_ind)| {
            let zero = F::ZERO;
            let f2_val = lhs_data
                .f
                .get((idx * (1 << (lhs_data.num_vars() - num_dataparallel_bits)) + x_ind) as usize)
                .unwrap_or(zero);
            let f3_val = rhs_data
                .f
                .get((idx * (1 << (rhs_data.num_vars() - num_dataparallel_bits)) + y_ind) as usize)
                .unwrap_or(zero);
            res_table[(num_gate_outputs_per_dataparallel_instance * idx + z_ind) as usize] +=
                gate_operation.perform_operation(f2_val, f3_val);
        });
    });

    MultilinearExtension::new(res_table)
}

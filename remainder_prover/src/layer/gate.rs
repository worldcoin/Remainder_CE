//! module for defining the gate layer, uses the libra trick
//! to reduce the number of rounds for gate layers (with binary operations)

/// Helper functions used in the gate sumcheck algorithms.
pub mod gate_helpers;
mod new_interface_tests;

use std::{
    cmp::{max, Ordering},
    collections::HashSet,
};

use gate_helpers::bind_round_gate;
use itertools::Itertools;
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{Claim, ClaimError, RawClaim},
    layer::{
        product::{PostSumcheckLayer, Product},
        Layer, LayerError, LayerId, VerificationError,
    },
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, verifier_mle::VerifierMle, Mle, MleIndex,
    },
    prover::{global_config::global_prover_lazy_beta_evals, SumcheckProof},
    sumcheck::{evaluate_at_a_point, SumcheckEvals},
};

use anyhow::{anyhow, Ok, Result};

pub use self::gate_helpers::{
    check_fully_bound, compute_full_gate, compute_sumcheck_message_gate,
    compute_sumcheck_message_no_beta_table, compute_sumcheck_messages_data_parallel_gate,
    index_mle_indices_gate, prove_round_dataparallel_phase, GateError,
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
#[serde(bound = "F: Field")]
pub struct GateLayer<F: Field> {
    /// The layer id associated with this gate layer.
    pub layer_id: LayerId,
    /// The number of bits representing the number of "dataparallel" copies of the circuit.
    pub num_dataparallel_vars: usize,
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
    u_challenges: Vec<F>,
    /// the beta table which enumerates the incoming claim's challenge points on the MLE
    beta_g1: Option<MultilinearExtension<F>>,
    /// the beta table which enumerates the incoming claim's challenge points on the
    /// dataparallel vars of the MLE
    beta_g2: Option<MultilinearExtension<F>>,
    /// the incoming claim's challenge points on the MLE
    g1: Option<Vec<F>>,
    /// the incoming claim's challenge points on the dataparallel vars of the MLE
    g2: Option<Vec<F>>,
    // the number of rounds in phase 1
    num_rounds_phase1: usize,
}

impl<F: Field> Layer<F> for GateLayer<F> {
    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn prove(
        &mut self,
        claim: RawClaim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<()> {
        let mut sumcheck_rounds = vec![];
        let (mut beta_g1, mut beta_g2) = self.compute_beta_tables(claim.get_point());
        let mut beta_g2_fully_bound = F::ONE;
        // We perform the dataparallel initialization only if there is at least one variable
        // representing which copy we are in.
        if self.num_dataparallel_vars > 0 {
            let (dataparallel_rounds, beta_g2_bound) = self
                .perform_dataparallel_phase(&mut beta_g1, &mut beta_g2, transcript_writer)
                .unwrap();
            beta_g2_fully_bound = beta_g2_bound;
            sumcheck_rounds.extend(dataparallel_rounds.0);
        }
        // We perform the rounds binding "x" variables (phase 1) and the rounds binding "y" variables (phase 2) in sequence.
        let (phase_1_rounds, f2_at_u, u_challenges) = self
            .perform_phase_1(
                claim.get_point()[self.num_dataparallel_vars..].to_vec(),
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

        // Finally, send the claimed values for each of the bound MLEs to the verifier
        // First, send the claimed value of V_{i + 1}(g_2, u)
        let lhs_reduced = &self.phase_1_mles.as_ref().unwrap()[0][1];
        let rhs_reduced = &self.phase_2_mles.as_ref().unwrap()[0][1];
        transcript_writer.append("Evaluation of V_{i + 1}(g_2, u)", lhs_reduced.value());
        // Next, send the claimed value of V_{i + 1}(g_2, v)
        transcript_writer.append("Evaluation of V_{i + 1}(g_2, v)", rhs_reduced.value());

        Ok(())
    }

    fn initialize(&mut self, claim_point: &[F]) -> Result<()> {
        if !global_prover_lazy_beta_evals() {
            let beta_g1 = BetaValues::new_beta_equality_mle(
                claim_point[self.num_dataparallel_vars..].to_vec(),
            );
            self.set_beta_g1(beta_g1);

            let beta_g2 = BetaValues::new_beta_equality_mle(
                claim_point[..self.num_dataparallel_vars].to_vec(),
            );
            self.set_beta_g2(beta_g2);
        } else {
            self.set_g1(claim_point[self.num_dataparallel_vars..].to_vec());
            self.set_g2(claim_point[..self.num_dataparallel_vars].to_vec());
        }

        self.lhs.index_mle_indices(0);
        self.rhs.index_mle_indices(0);

        Ok(())
    }

    fn compute_round_sumcheck_message(&mut self, round_index: usize) -> Result<Vec<F>> {
        // TODO!(ende): right now we still initializes the beta even the LAZY_BETA_EVALUATION flag is on
        // it's because fn `compute_sumcheck_messages_data_parallel_identity_gate` cannot lazy
        // evaluate beta's within it yet
        if round_index == 0 && global_prover_lazy_beta_evals() {
            let (beta_g2, beta_g1) = (
                BetaValues::new_beta_equality_mle(self.g2.as_ref().unwrap().clone()),
                BetaValues::new_beta_equality_mle(self.g1.as_ref().unwrap().clone()),
            );
            self.set_beta_g1(beta_g1);
            self.set_beta_g2(beta_g2);
        }

        let rounds_before_phase_2 = self.num_dataparallel_vars + self.num_rounds_phase1;

        match round_index.cmp(&self.num_dataparallel_vars) {
            // dataparallel phase
            Ordering::Less => Ok(compute_sumcheck_messages_data_parallel_gate(
                &self.lhs,
                &self.rhs,
                self.beta_g2.as_ref().unwrap(),
                self.beta_g1.as_ref().unwrap(),
                self.gate_operation,
                &self.nonzero_gates,
                self.num_dataparallel_vars - round_index,
            )
            .unwrap()),

            // init phase 1
            Ordering::Equal => {
                let num_x = self.lhs.num_free_vars();

                // Because we are binding `x` variables after this phase, all bookkeeping tables should have size
                // 2^(number of x variables).
                let mut a_hg_rhs = vec![F::ZERO; 1 << num_x];
                let mut a_hg_lhs = vec![F::ZERO; 1 << num_x];

                // Over here, we are looping through the nonzero gates using the Libra trick. This takes advantage
                // of the sparsity of the gate function. if we have the following expression:
                // f1(z, x, y)(f2(x) + f3(y)) then because we are only binding the "x" variables, we can simply
                // distribute over the y variables and construct bookkeeping tables that are size 2^(num_x_variables).
                self.nonzero_gates.iter().for_each(|(z_ind, x_ind, y_ind)| {
                    let beta_g_at_z = if global_prover_lazy_beta_evals() {
                        BetaValues::compute_beta_over_challenge_and_index(
                            self.g1.as_ref().unwrap(),
                            *z_ind,
                        )
                    } else {
                        self.beta_g1
                            .as_ref()
                            .unwrap()
                            .get(*z_ind)
                            .unwrap_or(F::ZERO)
                    };
                    let f_3_at_y = self.rhs.get(*y_ind).unwrap_or(F::ZERO);
                    a_hg_rhs[*x_ind] += beta_g_at_z * f_3_at_y;
                    if self.gate_operation == BinaryOperation::Add {
                        a_hg_lhs[*x_ind] += beta_g_at_z;
                    }
                });

                let a_hg_rhs_mle = DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0));

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
                            vec![a_hg_rhs_mle],
                        ]
                    }
                    BinaryOperation::Mul => {
                        vec![vec![a_hg_rhs_mle, self.lhs.clone()]]
                    }
                };

                phase_1_mles.iter_mut().for_each(|mle_vec| {
                    index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
                });

                self.set_phase_1(phase_1_mles.clone());

                let max_deg = phase_1_mles
                    .iter()
                    .fold(0, |acc, elem| max(acc, elem.len()));

                let init_mles: Vec<Vec<&DenseMle<F>>> = phase_1_mles
                    .iter()
                    .map(|mle_vec| {
                        let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                        mleerences
                    })
                    .collect();
                let evals_vec = init_mles
                    .iter()
                    .map(|mle_vec| {
                        compute_sumcheck_message_no_beta_table(
                            mle_vec,
                            self.num_dataparallel_vars,
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
                let SumcheckEvals(mut final_vec_evals) = final_evals;

                let beta_g2_fully_bound = self.beta_g2.as_ref().unwrap().value();

                final_vec_evals
                    .iter_mut()
                    .for_each(|eval| *eval *= beta_g2_fully_bound);

                Ok(final_vec_evals)
            }

            Ordering::Greater => match round_index.cmp(&rounds_before_phase_2) {
                // phase 1
                Ordering::Less => {
                    let phase_1_mles: Vec<Vec<&DenseMle<F>>> = self
                        .phase_1_mles
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|mle_vec| {
                            let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                            mleerences
                        })
                        .collect();

                    let max_deg = phase_1_mles
                        .iter()
                        .fold(0, |acc, elem| max(acc, elem.len()));
                    let evals_vec = phase_1_mles
                        .iter()
                        .map(|mle_vec| {
                            compute_sumcheck_message_no_beta_table(mle_vec, round_index, max_deg)
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
                    let SumcheckEvals(mut final_vec_evals) = final_evals;

                    let beta_g2_fully_bound = self.beta_g2.as_ref().unwrap().value();

                    final_vec_evals
                        .iter_mut()
                        .for_each(|eval| *eval *= beta_g2_fully_bound);

                    Ok(final_vec_evals)
                }

                // init phase 2
                Ordering::Equal => {
                    let u_claim = &self.u_challenges;
                    assert_eq!(self.u_challenges.len(), self.num_rounds_phase1);

                    let f2 = &self.phase_1_mles.as_ref().unwrap()[0][1];
                    let f2_at_u = f2.value();

                    let beta_g1 = self.beta_g1.as_ref().unwrap();

                    // Create a beta table according to the challenges used to bind the x variables.
                    let beta_u = BetaValues::new_beta_equality_mle(u_claim.to_vec());
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
                            let gz = beta_g1.get(z_ind).unwrap_or(F::ZERO);
                            let ux = beta_u.get(x_ind).unwrap_or(F::ZERO);
                            let adder = gz * ux;
                            a_f1_lhs[y_ind] += adder * f2_at_u;
                            if self.gate_operation == BinaryOperation::Add {
                                a_f1_rhs[y_ind] += adder;
                            }
                        });

                    let a_f1_lhs_mle = DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0));

                    // We need to multiply h_g(x) by f_2(x)
                    let mut phase_2_mles = match self.gate_operation {
                        BinaryOperation::Add => {
                            vec![
                                vec![
                                    DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0)),
                                    self.rhs.clone(),
                                ],
                                vec![a_f1_lhs_mle],
                            ]
                        }
                        BinaryOperation::Mul => {
                            vec![vec![a_f1_lhs_mle, self.rhs.clone()]]
                        }
                    };

                    phase_2_mles.iter_mut().for_each(|mle_vec| {
                        index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
                    });
                    self.set_phase_2(phase_2_mles.clone());

                    // Return the first sumcheck message of this phase.
                    let max_deg = phase_2_mles
                        .iter()
                        .fold(0, |acc, elem| max(acc, elem.len()));

                    let init_mles: Vec<Vec<&DenseMle<F>>> = phase_2_mles
                        .iter()
                        .map(|mle_vec| {
                            let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                            mleerences
                        })
                        .collect();
                    let evals_vec = init_mles
                        .iter()
                        .map(|mle_vec| {
                            compute_sumcheck_message_no_beta_table(
                                mle_vec,
                                self.num_dataparallel_vars,
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
                    let SumcheckEvals(mut final_vec_evals) = final_evals;

                    let beta_g2_fully_bound = self.beta_g2.as_ref().unwrap().value();

                    final_vec_evals
                        .iter_mut()
                        .for_each(|eval| *eval *= beta_g2_fully_bound);

                    Ok(final_vec_evals)
                }

                // phase 2
                Ordering::Greater => {
                    if self.phase_2_mles.as_ref().unwrap()[0][1].num_free_vars() > 0 {
                        let phase_2_mles: Vec<Vec<&DenseMle<F>>> = self
                            .phase_2_mles
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|mle_vec| {
                                let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                                mleerences
                            })
                            .collect();

                        let max_deg = phase_2_mles
                            .iter()
                            .fold(0, |acc, elem| max(acc, elem.len()));

                        let evals_vec = phase_2_mles
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
                        let SumcheckEvals(mut final_vec_evals) = final_evals;

                        let beta_g2_fully_bound = self.beta_g2.as_ref().unwrap().value();

                        final_vec_evals
                            .iter_mut()
                            .for_each(|eval| *eval *= beta_g2_fully_bound);

                        Ok(final_vec_evals)
                    } else {
                        Ok(vec![])
                    }
                }
            },
        }
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<()> {
        if round_index < self.num_dataparallel_vars {
            self.beta_g2.as_mut().unwrap().fix_variable(challenge);
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
            self.add_to_u_challenges(challenge);
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
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> super::product::PostSumcheckLayer<F, F> {
        let lhs_mle = &self.phase_1_mles.as_ref().unwrap()[0][1];
        let rhs_mle = &self.phase_2_mles.as_ref().unwrap()[0][1];

        let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
        let g1_challenges = claim_challenges[self.num_dataparallel_vars..].to_vec();

        let dataparallel_sumcheck_challenges =
            round_challenges[..self.num_dataparallel_vars].to_vec();
        let first_u_challenges = round_challenges
            [self.num_dataparallel_vars..self.num_dataparallel_vars + self.num_rounds_phase1]
            .to_vec();
        assert_eq!(first_u_challenges, self.u_challenges);
        let last_v_challenges =
            round_challenges[self.num_dataparallel_vars + self.num_rounds_phase1..].to_vec();
        // Compute the gate function bound at those variables.
        // Beta table corresponding to the equality of binding the x variables to u.
        let beta_u = BetaValues::new_beta_equality_mle(first_u_challenges);
        // Beta table corresponding to the equality of binding the y variables to v.
        let beta_v = BetaValues::new_beta_equality_mle(last_v_challenges);
        // Beta table representing all "z" label challenges.
        let beta_g = BetaValues::new_beta_equality_mle(g1_challenges);
        // Multiply the corresponding entries of the beta tables to get the full value of the gate function
        // i.e. f1(z, x, y) bound at the challenges f1(g1, u, v).
        let f_1_uv = self
            .nonzero_gates
            .iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                let gz = beta_g.get(*z_ind).unwrap_or(F::ZERO);
                let ux = beta_u.get(*x_ind).unwrap_or(F::ZERO);
                let vy = beta_v.get(*y_ind).unwrap_or(F::ZERO);
                acc + gz * ux * vy
            });

        let beta_bound = if self.num_dataparallel_vars != 0 {
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &dataparallel_sumcheck_challenges,
            )
        } else {
            F::ONE
        };

        match self.gate_operation {
            BinaryOperation::Add => PostSumcheckLayer(vec![
                Product::<F, F>::new(&[lhs_mle.clone()], f_1_uv * beta_bound),
                Product::<F, F>::new(&[rhs_mle.clone()], f_1_uv * beta_bound),
            ]),
            BinaryOperation::Mul => PostSumcheckLayer(vec![Product::<F, F>::new(
                &[lhs_mle.clone(), rhs_mle.clone()],
                f_1_uv * beta_bound,
            )]),
        }
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
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
        let val = lhs_reduced.value();
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_u,
            val,
            self.layer_id(),
            self.lhs.layer_id(),
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
        let val = rhs_reduced.value();
        let claim: Claim<F> = Claim::new(
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
#[serde(bound = "F: Field")]
pub struct GateLayerDescription<F: Field> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// The gate operation representing the fan-in-two relationship.
    gate_operation: BinaryOperation,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x, y) where the gate at label z is
    /// the output of performing an operation on gates with labels x and y.
    nonzero_gates: Vec<(usize, usize, usize)>,

    /// The left side of the expression, i.e. the mle that makes up the "x"
    /// variables.
    lhs_mle: MleDescription<F>,

    /// The mles that are constructed when initializing phase 2 (binding the y
    /// variables).
    rhs_mle: MleDescription<F>,

    /// The number of bits representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_vars: usize,
}

impl<F: Field> GateLayerDescription<F> {
    /// Constructor for a [GateLayerDescription].
    pub fn new(
        num_dataparallel_vars: Option<usize>,
        wiring: Vec<(usize, usize, usize)>,
        lhs_circuit_mle: MleDescription<F>,
        rhs_circuit_mle: MleDescription<F>,
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

impl<F: Field> LayerDescription<F> for GateLayerDescription<F> {
    type VerifierLayer = VerifierGateLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn verify_rounds(
        &self,
        claim: RawClaim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>> {
        // Storing challenges for the sake of claim generation later
        let mut challenges = vec![];

        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED
        // BITS
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;

        // Store all prover sumcheck messages to check against
        let mut sumcheck_messages: Vec<Vec<F>> = vec![];

        // First round check against the claim.
        let first_round_num_evals = match (self.gate_operation, self.num_dataparallel_vars) {
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
        if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1] != claim.get_eval() {
            return Err(anyhow!(VerificationError::SumcheckStartFailed));
        }

        // Check each of the messages -- note that here the verifier doesn't actually see the difference
        // between dataparallel rounds, phase 1 rounds, and phase 2 rounds; instead, the prover's proof reads
        // as a single continuous proof.
        for sumcheck_round_idx in 1..self.num_dataparallel_vars + num_u + num_v {
            // Read challenge r_{i - 1} from transcript
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
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
                .consume_elements("Sumcheck evaluations", univariate_num_evals)
                .unwrap();

            // Check: g_i(0) + g_i(1) =? g_{i - 1}(r_{i - 1})
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(anyhow!(VerificationError::SumcheckFailed));
            };

            // Add the prover message to the sumcheck messages
            sumcheck_messages.push(curr_evals);
            // Add the challenge.
            challenges.push(challenge);
        }

        // Final round of sumcheck -- sample r_n from transcript.
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        // Create the resulting verifier layer for claim tracking
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
            return Err(anyhow!(VerificationError::FinalSumcheckFailed));
        }

        Ok(VerifierLayerEnum::Gate(verifier_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        (0..num_u + num_v + self.num_dataparallel_vars).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer> {
        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED
        // BITS
        let num_u = self.lhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
            acc + match idx {
                MleIndex::Fixed(_) => 0,
                _ => 1,
            }
        }) - self.num_dataparallel_vars;
        let num_v = self.rhs_mle.var_indices().iter().fold(0_usize, |acc, idx| {
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
            claim_challenge_points: claim_point.to_vec(),
            dataparallel_sumcheck_challenges: dataparallel_challenges,
            first_u_challenges,
            last_v_challenges,
        };

        Ok(verifier_gate_layer)
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> super::product::PostSumcheckLayer<F, Option<F>> {
        let num_rounds_phase1 = self.lhs_mle.num_free_vars() - self.num_dataparallel_vars;

        let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
        let g1_challenges = claim_challenges[self.num_dataparallel_vars..].to_vec();

        let dataparallel_sumcheck_challenges =
            round_challenges[..self.num_dataparallel_vars].to_vec();
        let first_u_challenges = round_challenges
            [self.num_dataparallel_vars..self.num_dataparallel_vars + num_rounds_phase1]
            .to_vec();
        let last_v_challenges =
            round_challenges[self.num_dataparallel_vars + num_rounds_phase1..].to_vec();
        // Compute the gate function bound at those variables.
        // Beta table corresponding to the equality of binding the x variables to u.
        let beta_u = BetaValues::new_beta_equality_mle(first_u_challenges);
        // Beta table corresponding to the equality of binding the y variables to v.
        let beta_v = BetaValues::new_beta_equality_mle(last_v_challenges);
        // Beta table representing all "z" label challenges.
        let beta_g = BetaValues::new_beta_equality_mle(g1_challenges);
        // Multiply the corresponding entries of the beta tables to get the full value of the gate function
        // i.e. f1(z, x, y) bound at the challenges f1(g1, u, v).
        let f_1_uv = self
            .nonzero_gates
            .iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                let gz = beta_g.get(*z_ind).unwrap_or(F::ZERO);
                let ux = beta_u.get(*x_ind).unwrap_or(F::ZERO);
                let vy = beta_v.get(*y_ind).unwrap_or(F::ZERO);
                acc + gz * ux * vy
            });

        let beta_bound = if self.num_dataparallel_vars != 0 {
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &dataparallel_sumcheck_challenges,
            )
        } else {
            F::ONE
        };

        let lhs_challenges = &round_challenges[..self.num_dataparallel_vars + num_rounds_phase1];
        let rhs_challenges = &round_challenges[..self.num_dataparallel_vars]
            .iter()
            .copied()
            .chain(round_challenges[self.num_dataparallel_vars + num_rounds_phase1..].to_vec())
            .collect_vec();

        match self.gate_operation {
            BinaryOperation::Add => PostSumcheckLayer(vec![
                Product::<F, Option<F>>::new(
                    &[self.lhs_mle.clone()],
                    f_1_uv * beta_bound,
                    lhs_challenges,
                ),
                Product::<F, Option<F>>::new(
                    &[self.rhs_mle.clone()],
                    f_1_uv * beta_bound,
                    rhs_challenges,
                ),
            ]),
            BinaryOperation::Mul => {
                PostSumcheckLayer(vec![Product::<F, Option<F>>::new_from_mul_gate(
                    &[self.lhs_mle.clone(), self.rhs_mle.clone()],
                    f_1_uv * beta_bound,
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

    fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        vec![&self.lhs_mle, &self.rhs_mle]
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
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
        mle_outputs_necessary: &HashSet<&MleDescription<F>>,
        circuit_map: &mut CircuitMap<F>,
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
        let mut res_table = vec![F::ZERO; res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            self.nonzero_gates.iter().for_each(|(z_ind, x_ind, y_ind)| {
                let zero = F::ZERO;
                let f2_val = lhs_data
                    .f
                    .get(idx * (1 << (lhs_data.num_vars() - self.num_dataparallel_vars)) + x_ind)
                    .unwrap_or(zero);
                let f3_val = rhs_data
                    .f
                    .get(idx * (1 << (rhs_data.num_vars() - self.num_dataparallel_vars)) + y_ind)
                    .unwrap_or(zero);
                res_table[num_gate_outputs_per_dataparallel_instance * idx + z_ind] +=
                    self.gate_operation.perform_operation(f2_val, f3_val);
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
    pub fn evaluate(&self, claim: &RawClaim<F>) -> F {
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
            .iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                let gz = beta_g.get(*z_ind).unwrap_or(F::ZERO);
                let ux = beta_u.get(*x_ind).unwrap_or(F::ZERO);
                let vy = beta_v.get(*y_ind).unwrap_or(F::ZERO);
                acc + gz * ux * vy
            });

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

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
        // Grab the claim on the left side.
        // TODO!(ryancao): Do error handling here!
        let lhs_vars = self.lhs_mle.var_indices();
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

        let lhs_claim: Claim<F> =
            Claim::new(lhs_point, lhs_val, self.layer_id(), self.lhs_mle.layer_id());

        // Grab the claim on the right side.
        // TODO!(ryancao): Do error handling here!
        let rhs_vars: &[MleIndex<F>] = self.rhs_mle.var_indices();
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

        let rhs_claim: Claim<F> =
            Claim::new(rhs_point, rhs_val, self.layer_id(), self.rhs_mle.layer_id());

        Ok(vec![lhs_claim, rhs_claim])
    }
}

impl<F: Field> GateLayer<F> {
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
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMle<F>,
        rhs: DenseMle<F>,
        gate_operation: BinaryOperation,
        layer_id: LayerId,
    ) -> Self {
        let num_dataparallel_vars = num_dataparallel_vars.unwrap_or(0);
        let num_rounds_phase1 = lhs.num_free_vars() - num_dataparallel_vars;

        GateLayer {
            num_dataparallel_vars,
            nonzero_gates,
            lhs,
            rhs,
            layer_id,
            phase_1_mles: None,
            phase_2_mles: None,
            gate_operation,
            u_challenges: vec![],
            beta_g1: None,
            beta_g2: None,
            g1: None,
            g2: None,
            num_rounds_phase1,
        }
    }

    fn set_beta_g1(&mut self, beta_g1: MultilinearExtension<F>) {
        self.beta_g1 = Some(beta_g1);
    }

    fn set_beta_g2(&mut self, beta_g2: MultilinearExtension<F>) {
        self.beta_g2 = Some(beta_g2);
    }

    fn set_g1(&mut self, g1: Vec<F>) {
        self.g1 = Some(g1);
    }

    fn set_g2(&mut self, g2: Vec<F>) {
        self.g2 = Some(g2);
    }

    /// bookkeeping tables necessary for binding x
    fn set_phase_1(&mut self, mles: Vec<Vec<DenseMle<F>>>) {
        self.phase_1_mles = Some(mles);
    }

    /// bookkeeping tables necessary for binding y
    fn set_phase_2(&mut self, mles: Vec<Vec<DenseMle<F>>>) {
        self.phase_2_mles = Some(mles);
    }

    /// adding to all the u challenges
    fn add_to_u_challenges(&mut self, u_challenge: F) {
        self.u_challenges.push(u_challenge);
    }

    fn compute_beta_tables(
        &mut self,
        challenges: &[F],
    ) -> (MultilinearExtension<F>, MultilinearExtension<F>) {
        let mut g2_challenges = vec![];
        let mut g1_challenges = vec![];

        challenges
            .iter()
            .enumerate()
            .for_each(|(bit_idx, challenge)| {
                if bit_idx < self.num_dataparallel_vars {
                    g2_challenges.push(*challenge);
                } else {
                    g1_challenges.push(*challenge);
                }
            });

        // Create two separate beta tables for each, as they are handled differently.
        let beta_g2 = BetaValues::new_beta_equality_mle(g2_challenges);
        let beta_g1 = BetaValues::new_beta_equality_mle(g1_challenges);

        (beta_g1, beta_g2)
    }

    /// Initialize the dataparallel phase: construct the necessary mles and return the first sumcheck message.
    /// This will then set the necessary fields of the [Gate] struct so that the dataparallel bits can be
    /// correctly bound during the first `num_dataparallel_vars` rounds of sumcheck.
    fn init_dataparallel_phase(
        &mut self,
        beta_g1: &mut MultilinearExtension<F>,
        beta_g2: &mut MultilinearExtension<F>,
    ) -> Result<Vec<F>> {
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
            self.num_dataparallel_vars,
        )
    }

    /// Initialize phase 1, or the necessary mles in order to bind the variables in the `lhs` of the
    /// expression. Once this phase is initialized, the sumcheck rounds binding the "x" variables can
    /// be performed.
    fn init_phase_1(&mut self, challenges: Vec<F>) -> Result<Vec<F>> {
        let beta_g1 = BetaValues::new_beta_equality_mle(challenges);

        let num_x = self.lhs.num_free_vars();

        // Because we are binding `x` variables after this phase, all bookkeeping tables should have size
        // 2^(number of x variables).
        let mut a_hg_rhs = vec![F::ZERO; 1 << num_x];
        let mut a_hg_lhs = vec![F::ZERO; 1 << num_x];

        // Over here, we are looping through the nonzero gates using the Libra trick. This takes advantage
        // of the sparsity of the gate function. if we have the following expression:
        // f1(z, x, y)(f2(x) + f3(y)) then because we are only binding the "x" variables, we can simply
        // distribute over the y variables and construct bookkeeping tables that are size 2^(num_x_variables).
        self.nonzero_gates.iter().for_each(|(z_ind, x_ind, y_ind)| {
            let beta_g_at_z = beta_g1.get(*z_ind).unwrap_or(F::ZERO);
            let f_3_at_y = self.rhs.get(*y_ind).unwrap_or(F::ZERO);
            a_hg_rhs[*x_ind] += beta_g_at_z * f_3_at_y;
            if self.gate_operation == BinaryOperation::Add {
                a_hg_lhs[*x_ind] += beta_g_at_z;
            }
        });

        let a_hg_rhs_mle = DenseMle::new_from_raw(a_hg_rhs, LayerId::Input(0));

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
                    vec![a_hg_rhs_mle],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![a_hg_rhs_mle, self.lhs.clone()]]
            }
        };

        phase_1_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });

        self.phase_1_mles = Some(phase_1_mles.clone());

        let max_deg = phase_1_mles
            .iter()
            .fold(0, |acc, elem| max(acc, elem.len()));

        let init_mles: Vec<Vec<&DenseMle<F>>> = phase_1_mles
            .iter()
            .map(|mle_vec| {
                let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                mleerences
            })
            .collect();
        let evals_vec = init_mles
            .iter()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_vars, max_deg)
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
        beta_g1: &MultilinearExtension<F>,
    ) -> Result<Vec<F>> {
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
                let gz = beta_g1.get(z_ind).unwrap_or(F::ZERO);
                let ux = beta_u.get(x_ind).unwrap_or(F::ZERO);
                let adder = gz * ux;
                a_f1_lhs[y_ind] += adder * f_at_u;
                if self.gate_operation == BinaryOperation::Add {
                    a_f1_rhs[y_ind] += adder;
                }
            });

        let a_f1_lhs_mle = DenseMle::new_from_raw(a_f1_lhs, LayerId::Input(0));

        // We need to multiply h_g(x) by f_2(x)
        let mut phase_2_mles = match self.gate_operation {
            BinaryOperation::Add => {
                vec![
                    vec![
                        DenseMle::new_from_raw(a_f1_rhs, LayerId::Input(0)),
                        self.rhs.clone(),
                    ],
                    vec![a_f1_lhs_mle],
                ]
            }
            BinaryOperation::Mul => {
                vec![vec![a_f1_lhs_mle, self.rhs.clone()]]
            }
        };

        phase_2_mles.iter_mut().for_each(|mle_vec| {
            index_mle_indices_gate(mle_vec, self.num_dataparallel_vars);
        });
        self.phase_2_mles = Some(phase_2_mles.clone());

        // Return the first sumcheck message of this phase.
        let max_deg = phase_2_mles
            .iter()
            .fold(0, |acc, elem| max(acc, elem.len()));

        let init_mles: Vec<Vec<&DenseMle<F>>> = phase_2_mles
            .iter()
            .map(|mle_vec| {
                let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                mleerences
            })
            .collect();
        let evals_vec = init_mles
            .iter()
            .map(|mle_vec| {
                compute_sumcheck_message_no_beta_table(mle_vec, self.num_dataparallel_vars, max_deg)
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
        beta_g1: &mut MultilinearExtension<F>,
        beta_g2: &mut MultilinearExtension<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(SumcheckProof<F>, F)> {
        // Initialization, first message comes from here.
        let mut challenges: Vec<F> = vec![];

        let first_message = self.init_dataparallel_phase(beta_g1, beta_g2).expect(
            "could not evaluate original lhs and rhs in order to get first sumcheck message",
        );

        let (lhs, rhs) = (&mut self.lhs, &mut self.rhs);

        transcript_writer
            .append_elements("Initial Sumcheck evaluations DATAPARALLEL", &first_message);
        let num_rounds_copy_phase = self.num_dataparallel_vars;

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
                    self.num_dataparallel_vars - round,
                    self.gate_operation,
                )
                .unwrap();
                transcript_writer.append_elements("Sumcheck evaluations DATAPARALLEL", &eval);
                Ok(eval)
            }))
            .try_collect()?;

        // Bind the final challenge, update the final beta table.
        let final_chal_copy =
            transcript_writer.get_challenge("Final Sumcheck challenge DATAPARALLEL");
        // Fix the variable and everything as you would in the last round of sumcheck
        // the evaluations from this is what you return from the first round of sumcheck in the next phase!
        beta_g2.fix_variable(final_chal_copy);
        self.lhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);
        self.rhs
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        let beta_g2_fully_bound = beta_g2.value();
        Ok((sumcheck_rounds.into(), beta_g2_fully_bound))
    }

    // We are binding the "x" variables of the `lhs`. At the end of this, the lhs of the expression
    // assuming we have a fan-in-two gate must be fully bound.
    fn perform_phase_1(
        &mut self,
        challenge: Vec<F>,
        beta_g2_fully_bound: F,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(SumcheckProof<F>, F, Vec<F>)> {
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
                bind_round_gate(round + self.num_dataparallel_vars, challenge, phase_1_mles);
                let phase_1_mles: Vec<Vec<&DenseMle<F>>> = phase_1_mles
                    .iter()
                    .map(|mle_vec| {
                        let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                        mleerences
                    })
                    .collect();
                let eval = compute_sumcheck_message_gate(
                    round + self.num_dataparallel_vars,
                    &phase_1_mles,
                )
                .into_iter()
                .map(|eval| eval * beta_g2_fully_bound)
                .collect_vec();
                transcript_writer.append_elements("Sumcheck evaluations PHASE 1", &eval);
                Ok(eval)
            }))
            .try_collect()?;

        // Final challenge after binding x (left side of the sum).
        let final_chal_u =
            transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
        challenges.push(final_chal_u);

        phase_1_mles.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.fix_variable(
                    num_rounds_phase1 - 1 + self.num_dataparallel_vars,
                    final_chal_u,
                );
            })
        });

        let f_2 = phase_1_mles[0][1].clone();

        let f2_at_u = f_2.value();
        Ok((sumcheck_rounds.into(), f2_at_u, challenges))
    }

    // These are the rounds binding the "y" variables of the expression. At the end of this, the entire
    // expression is fully bound because this is the last phase in proving the gate layer.
    fn perform_phase_2(
        &mut self,
        f_at_u: F,
        phase_1_challenges: Vec<F>,
        beta_g1: MultilinearExtension<F>,
        beta_g2_fully_bound: F,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<SumcheckProof<F>> {
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
                    bind_round_gate(round + self.num_dataparallel_vars, challenge, phase_2_mles);
                    let phase_2_mles: Vec<Vec<&DenseMle<F>>> = phase_2_mles
                        .iter()
                        .map(|mle_vec| {
                            let mleerences: Vec<&DenseMle<F>> = mle_vec.iter().collect();
                            mleerences
                        })
                        .collect();
                    let eval = compute_sumcheck_message_gate(
                        round + self.num_dataparallel_vars,
                        &phase_2_mles,
                    )
                    .into_iter()
                    .map(|eval| eval * beta_g2_fully_bound)
                    .collect_vec();
                    transcript_writer.append_elements("Sumcheck evaluations", &eval);
                    Ok(eval)
                }))
                .try_collect()?;

            // Final round of sumcheck.
            let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge");
            challenges.push(final_chal);

            phase_2_mles.iter_mut().for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.fix_variable(
                        num_rounds_phase2 - 1 + self.num_dataparallel_vars,
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
    wiring: Vec<(usize, usize, usize)>,
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

    let mut res_table = vec![F::ZERO; res_table_num_entries];
    // TDH(ende): investigate if this can be parallelized (and if it's a bottleneck)
    (0..num_dataparallel_vals).for_each(|idx| {
        wiring.iter().for_each(|(z_ind, x_ind, y_ind)| {
            let zero = F::ZERO;
            let f2_val = lhs_data
                .f
                .get(idx * (1 << (lhs_data.num_vars() - num_dataparallel_bits)) + x_ind)
                .unwrap_or(zero);
            let f3_val = rhs_data
                .f
                .get(idx * (1 << (rhs_data.num_vars() - num_dataparallel_bits)) + y_ind)
                .unwrap_or(zero);
            res_table[num_gate_outputs_per_dataparallel_instance * idx + z_ind] +=
                gate_operation.perform_operation(f2_val, f3_val);
        });
    });

    MultilinearExtension::new(res_table)
}

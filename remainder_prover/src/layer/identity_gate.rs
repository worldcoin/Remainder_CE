//! Identity gate id(z, x) determines whether the xth gate from the
//! i + 1th layer contributes to the zth gate in the ith layer.

use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{Debug, Formatter},
};

use crate::{
    claims::{Claim, ClaimError, RawClaim},
    layer::{LayerError, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, verifier_mle::VerifierMle, Mle, MleIndex,
    },
    sumcheck::*,
};
use itertools::Itertools;
use remainder_shared_types::{
    config::global_config::{global_prover_lazy_beta_evals, global_verifier_lazy_beta_evals},
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use thiserror::Error;

use super::{
    gate::gate_helpers::evaluate_mle_product_no_beta_table,
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    Layer, LayerDescription, LayerId, VerifierLayer,
};

use anyhow::{anyhow, Ok, Result};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// The circuit Description for an [IdentityGate].
#[derive(Serialize, Deserialize, Clone, Hash)]
#[serde(bound = "F: Field")]
pub struct IdentityGateLayerDescription<F: Field> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(u32, u32)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: MleDescription<F>,

    /// The total number of variables in the layer.
    total_num_vars: usize,

    /// The number of vars representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_vars: usize,
}

impl<F: Field> std::fmt::Debug for IdentityGateLayerDescription<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdentityGateLayerDescription")
            .field("id", &self.id)
            .field("wiring.len()", &self.wiring.len())
            .field("source_mle", &self.source_mle)
            .field("num_dataparallel_vars", &self.num_dataparallel_vars)
            .finish()
    }
}

impl<F: Field> IdentityGateLayerDescription<F> {
    /// Constructor for [IdentityGateLayerDescription].
    /// Arguments:
    /// * `id`: The layer id associated with this layer.
    /// * `source_mle`: The Mle that is being routed to this layer.
    /// * `nonzero_gates`: A list of tuples representing the gates that are nonzero, in the form `(dest_idx, src_idx)`.
    /// * `total_num_vars`: The total number of variables in the layer.
    /// * `num_dataparallel_vars`: The number of dataparallel variables to use in this layer.
    pub fn new(
        id: LayerId,
        wiring: Vec<(u32, u32)>,
        source_mle: MleDescription<F>,
        total_num_vars: usize,
        num_dataparallel_vars: Option<usize>,
    ) -> Self {
        Self {
            id,
            wiring,
            source_mle,
            total_num_vars,
            num_dataparallel_vars: num_dataparallel_vars.unwrap_or(0),
        }
    }
}

impl<F: Field> LayerDescription<F> for IdentityGateLayerDescription<F> {
    type VerifierLayer = VerifierIdentityGateLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn verify_rounds(
        &self,
        claim: RawClaim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>> {
        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
        let mut g_prev_round = vec![claim.get_eval()];

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = F::ZERO;

        let num_rounds = self.sumcheck_round_indices().len();

        // For round 1 <= i <= n, perform the check:
        for _round in 0..num_rounds {
            // Degree of independent variable is always quadratic!
            // (regardless of if there's dataparallel or not)
            // V_i(g_2, g_1) = \sum_{p_2} \sum_{x} \beta(g_2, p_2) f_1(g_1, x) (V_{i + 1}(p_2, x))
            let degree = 2;

            let g_cur_round = transcript_reader
                .consume_elements("Sumcheck message", degree + 1)
                .map_err(|_| VerificationError::TranscriptError)?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_challenge("Sumcheck challenge")?;

            // Verify that:
            //       `g_i(0) + g_i(1) == g_{i - 1}(r_{i-1})`
            let g_i_zero = evaluate_at_a_point(&g_cur_round, F::ZERO).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, F::ONE).unwrap();
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();

            if g_i_zero + g_i_one != g_prev_r_prev {
                dbg!(_round);
                return Err(anyhow!(VerificationError::SumcheckFailed));
            }

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // Evalute `g_n(r_n)`.
        // Note: If there were no nonlinear rounds, this value reduces to
        // `claim.get_result()` due to how we initialized `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        let verifier_id_gate_layer = self
            .convert_into_verifier_layer(&challenges, claim.get_point(), transcript_reader)
            .unwrap();
        let final_result = verifier_id_gate_layer.evaluate(&claim);

        if g_final_r_final != final_result {
            return Err(anyhow!(VerificationError::FinalSumcheckFailed));
        }

        Ok(VerifierLayerEnum::IdentityGate(verifier_id_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        let num_vars = self
            .source_mle
            .var_indices()
            .iter()
            .fold(0_usize, |acc, idx| {
                acc + match idx {
                    MleIndex::Fixed(_) => 0,
                    _ => 1,
                }
            });

        (0..num_vars).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_challenges: &[F],
        _claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer> {
        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED
        // vars
        let num_u = self
            .source_mle
            .var_indices()
            .iter()
            .fold(0_usize, |acc, idx| {
                acc + match idx {
                    MleIndex::Fixed(_) => 0,
                    _ => 1,
                }
            })
            - self.num_dataparallel_vars;

        // We want to separate the challenges into which ones are from the dataparallel vars,
        // which ones and are for binding x (phase 1)
        let mut sumcheck_bindings_vec = sumcheck_challenges.to_vec();
        let first_u_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_vars);
        let dataparallel_sumcheck_challenges = sumcheck_bindings_vec;

        assert_eq!(first_u_challenges.len(), num_u);

        // Since the original mles are dataparallel, the challenges are the concat of the copy vars and the variable bound vars.
        let src_verifier_mle = self
            .source_mle
            .into_verifier_mle(sumcheck_challenges, transcript_reader)
            .unwrap();

        // Create the resulting verifier layer for claim tracking
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_id_gate_layer = VerifierIdentityGateLayer {
            layer_id: self.layer_id(),
            wiring: self.wiring.clone(),
            source_mle: src_verifier_mle,
            first_u_challenges,
            total_num_vars: self.total_num_vars,
            num_dataparallel_rounds: self.num_dataparallel_vars,
            dataparallel_sumcheck_challenges,
        };

        Ok(verifier_id_gate_layer)
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        // TODO(ryancao): Distinguish between the prover and verifier here
        let beta_ug = if !global_prover_lazy_beta_evals() {
            Some((
                BetaValues::new_beta_equality_mle(
                    round_challenges[self.num_dataparallel_vars..].to_vec(),
                ),
                BetaValues::new_beta_equality_mle(
                    claim_challenges[self.num_dataparallel_vars..].to_vec(),
                ),
            ))
        } else {
            None
        };

        #[cfg(feature = "parallel")]
        let f_1_uv = self
            .wiring
            .par_iter()
            .fold(
                || F::ZERO,
                |acc, (z_ind, x_ind)| {
                    let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                        (
                            beta_g1.get(*z_ind as usize).unwrap_or(F::ZERO),
                            beta_u.get(*x_ind as usize).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                &claim_challenges[self.num_dataparallel_vars..],
                                *z_ind as usize,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                &round_challenges[self.num_dataparallel_vars..],
                                *x_ind as usize,
                            ),
                        )
                    };

                    acc + gz * ux
                },
            )
            .sum::<F>();

        #[cfg(not(feature = "parallel"))]
        let f_1_uv = self.wiring.iter().fold(F::ZERO, |acc, (z_ind, x_ind)| {
            let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                (
                    beta_g1.get(*z_ind as usize).unwrap_or(F::ZERO),
                    beta_u.get(*x_ind as usize).unwrap_or(F::ZERO),
                )
            } else {
                (
                    BetaValues::compute_beta_over_challenge_and_index(
                        &claim_challenges[self.num_dataparallel_vars..],
                        *z_ind as usize,
                    ),
                    BetaValues::compute_beta_over_challenge_and_index(
                        &round_challenges[self.num_dataparallel_vars..],
                        *x_ind as usize,
                    ),
                )
            };

            acc + gz * ux
        });

        let beta_bound = if self.num_dataparallel_vars != 0 {
            let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &round_challenges[..self.num_dataparallel_vars],
            )
        } else {
            F::ONE
        };

        PostSumcheckLayer(vec![Product::<F, Option<F>>::new(
            &[self.source_mle.clone()],
            f_1_uv * beta_bound,
            round_challenges,
        )])
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        vec![&self.source_mle]
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let source_mle = self.source_mle.into_dense_mle(circuit_map);
        let id_gate_layer = IdentityGate::new(
            self.layer_id(),
            self.wiring.clone(),
            source_mle,
            self.total_num_vars,
            self.num_dataparallel_vars,
        );
        id_gate_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.source_mle.index_mle_indices(start_index);
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&MleDescription<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();
        let source_mle_data = circuit_map
            .get_data_from_circuit_mle(&self.source_mle)
            .unwrap();

        let res_table_num_entries = 1 << self.total_num_vars;
        let num_entries_per_dataparallel_instance =
            1 << (self.total_num_vars - self.num_dataparallel_vars);
        let mut remap_table = vec![F::ZERO; res_table_num_entries];

        (0..(1 << self.num_dataparallel_vars)).for_each(|data_parallel_idx| {
            self.wiring.iter().for_each(|(dest_idx, src_idx)| {
                let id_val = source_mle_data
                    .f
                    .get(
                        data_parallel_idx
                            * (1 << (self.source_mle.num_free_vars() - self.num_dataparallel_vars))
                            + (*src_idx as usize),
                    )
                    .unwrap_or(F::ZERO);
                remap_table[num_entries_per_dataparallel_instance * data_parallel_idx
                    + (*dest_idx as usize)] = id_val;
            });
        });
        let output_data = MultilinearExtension::new(remap_table);
        assert_eq!(
            output_data.num_vars(),
            mle_output_necessary.var_indices().len()
        );

        circuit_map.add_node(CircuitLocation::new(self.layer_id(), vec![]), output_data);
    }
}

impl<F: Field> VerifierIdentityGateLayer<F> {
    /// Computes the oracle query's value for a given [IdentityGateVerifierLayer].
    pub fn evaluate(&self, claim: &RawClaim<F>) -> F {
        let g2_challenges = claim.get_point()[..self.num_dataparallel_rounds].to_vec();
        let g1_challenges = claim.get_point()[self.num_dataparallel_rounds..].to_vec();

        // TODO(ryancao): Is this function also used by the prover???
        let beta_ug = if !global_verifier_lazy_beta_evals() {
            Some((
                BetaValues::new_beta_equality_mle(self.first_u_challenges.clone()),
                BetaValues::new_beta_equality_mle(g1_challenges.clone()),
            ))
        } else {
            None
        };

        #[cfg(feature = "parallel")]
        let f_1_uv = self
            .wiring
            .par_iter()
            .fold(
                || F::ZERO,
                |acc, (z_ind, x_ind)| {
                    let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                        (
                            beta_g1.f.get(*z_ind as usize).unwrap_or(F::ZERO),
                            beta_u.f.get(*x_ind as usize).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                &g1_challenges,
                                *z_ind as usize,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                &self.first_u_challenges,
                                *x_ind as usize,
                            ),
                        )
                    };

                    acc + gz * ux
                },
            )
            .sum::<F>();

        #[cfg(not(feature = "parallel"))]
        let f_1_uv = self.wiring.iter().fold(F::ZERO, |acc, (z_ind, x_ind)| {
            let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                (
                    beta_g1.f.get(*z_ind as usize).unwrap_or(F::ZERO),
                    beta_u.f.get(*x_ind as usize).unwrap_or(F::ZERO),
                )
            } else {
                (
                    BetaValues::compute_beta_over_challenge_and_index(
                        &g1_challenges,
                        *z_ind as usize,
                    ),
                    BetaValues::compute_beta_over_challenge_and_index(
                        &self.first_u_challenges,
                        *x_ind as usize,
                    ),
                )
            };

            acc + gz * ux
        });

        let beta_bound = BetaValues::compute_beta_over_two_challenges(
            &g2_challenges,
            &self.dataparallel_sumcheck_challenges,
        );
        // get the fully evaluated "expression"
        beta_bound * f_1_uv * self.source_mle.value()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
/// The layer representing a fully bound [IdentityGate].
pub struct VerifierIdentityGateLayer<F: Field> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(u32, u32)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: VerifierMle<F>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<F>,

    /// The total number of variables in the layer.
    total_num_vars: usize,

    /// The number of dataparallel rounds.
    num_dataparallel_rounds: usize,

    /// The challenges for `p_2`, as derived from sumcheck.
    dataparallel_sumcheck_challenges: Vec<F>,
}

impl<F: Field> VerifierLayer<F> for VerifierIdentityGateLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
        // Grab the claim on the left side.
        let source_vars = self.source_mle.var_indices();
        let source_point = source_vars
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
        let source_val = self.source_mle.value();

        let source_claim: Claim<F> = Claim::new(
            source_point,
            source_val,
            self.layer_id(),
            self.source_mle.layer_id(),
        );

        Ok(vec![source_claim])
    }
}

/// The layer trait implementation for [IdentityGate], which has the proving
/// functionality as well as the modular functions for each round of sumcheck.
impl<F: Field> Layer<F> for IdentityGate<F> {
    fn prove(
        &mut self,
        claim: RawClaim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<()> {
        self.initialize(claim.get_point())?;
        (0..self.source_mle.num_free_vars()).for_each(|round_idx| {
            let sumcheck_message = self.compute_round_sumcheck_message(round_idx).unwrap();
            transcript_writer.append_elements("Round sumcheck message", &sumcheck_message);
            let challenge = transcript_writer.get_challenge("Sumcheck challenge");
            self.bind_round_variable(round_idx, challenge).unwrap();
        });
        self.append_leaf_mles_to_transcript(transcript_writer);
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn initialize(&mut self, claim_point: &[F]) -> Result<()> {
        let (g2_challenges, g1_challenges) = claim_point.split_at(self.num_dataparallel_vars);
        self.set_g1_challenges(g1_challenges.to_vec());
        self.set_g2_challenges(g2_challenges.to_vec());

        if self.num_dataparallel_vars > 0 {
            let beta_g2 = BetaValues::new(
                self.g2_challenges
                    .as_ref()
                    .unwrap()
                    .clone()
                    .into_iter()
                    .enumerate()
                    .collect(),
            );
            self.set_beta_g2(beta_g2);
            self.init_dataparallel_phase(self.g1_challenges.as_ref().unwrap().clone());
        }

        self.source_mle.index_mle_indices(0);
        Ok(())
    }

    fn compute_round_sumcheck_message(&mut self, round_index: usize) -> Result<Vec<F>> {
        match round_index.cmp(&self.num_dataparallel_vars) {
            // Dataparallel phase.
            Ordering::Less => {
                let sumcheck_message = self
                    .compute_sumcheck_message_data_parallel_identity_gate_beta_cascade(round_index)
                    .unwrap();
                Ok(sumcheck_message)
            }

            // Initialize phase 1.
            Ordering::Equal => {
                self.init_phase_1(self.g1_challenges.as_ref().unwrap().clone());

                let mles: Vec<&DenseMle<F>> =
                    vec![&self.a_hg_mle_phase_1.as_ref().unwrap(), &self.source_mle];
                let independent_variable = mles
                    .iter()
                    .map(|mle| mle.mle_indices().contains(&MleIndex::Indexed(round_index)))
                    .reduce(|acc, item| acc | item)
                    .unwrap();
                let unscaled_sumcheck_evals =
                    evaluate_mle_product_no_beta_table(&mles, independent_variable, mles.len())
                        .unwrap();

                let beta_g2_fully_bound = if self.num_dataparallel_vars > 0 {
                    self.beta_g2
                        .as_ref()
                        .unwrap()
                        .updated_values
                        .values()
                        .fold(F::ONE, |acc, val| acc * *val)
                } else {
                    F::ONE
                };

                let first_round_sumcheck_evals = unscaled_sumcheck_evals
                    .0
                    .iter()
                    .map(|unscaled_eval| *unscaled_eval * beta_g2_fully_bound)
                    .collect();
                Ok(first_round_sumcheck_evals)
            }

            // Phase 1.
            Ordering::Greater => {
                let mles: Vec<&DenseMle<F>> =
                    vec![&self.a_hg_mle_phase_1.as_ref().unwrap(), &self.source_mle];
                let independent_variable = mles
                    .iter()
                    .map(|mle| mle.mle_indices().contains(&MleIndex::Indexed(round_index)))
                    .reduce(|acc, item| acc | item)
                    .unwrap();
                let unscaled_sumcheck_evals =
                    evaluate_mle_product_no_beta_table(&mles, independent_variable, mles.len())
                        .unwrap();

                let beta_g2_fully_bound = if self.num_dataparallel_vars > 0 {
                    self.beta_g2
                        .as_ref()
                        .unwrap()
                        .updated_values
                        .values()
                        .fold(F::ONE, |acc, val| acc * *val)
                } else {
                    F::ONE
                };
                let sumcheck_evals = unscaled_sumcheck_evals
                    .0
                    .iter()
                    .map(|unscaled_eval| *unscaled_eval * beta_g2_fully_bound)
                    .collect();
                Ok(sumcheck_evals)
            }
        }
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<()> {
        if round_index < self.num_dataparallel_vars {
            self.beta_g2
                .as_mut()
                .unwrap()
                .beta_update(round_index, challenge);
            self.dataparallel_af2_mle
                .as_mut()
                .unwrap()
                .fix_variable(round_index, challenge);
            self.source_mle.fix_variable(round_index, challenge);
            Ok(())
        } else {
            if self.num_dataparallel_vars > 0 {
                assert!(self.beta_g2.as_ref().unwrap().unbound_values.is_empty());
            }
            let a_hg_mle = self.a_hg_mle_phase_1.as_mut().unwrap();

            [a_hg_mle, &mut self.source_mle].iter_mut().for_each(|mle| {
                mle.fix_variable(round_index, challenge);
            });
            Ok(())
        }
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.source_mle.num_free_vars()).collect_vec()
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        // TODO(ryancao): Distinguish between prover and verifier here
        let beta_ug = if !global_prover_lazy_beta_evals() {
            Some((
                BetaValues::new_beta_equality_mle(
                    round_challenges[self.num_dataparallel_vars..].to_vec(),
                ),
                BetaValues::new_beta_equality_mle(
                    claim_challenges[self.num_dataparallel_vars..].to_vec(),
                ),
            ))
        } else {
            None
        };

        #[cfg(feature = "parallel")]
        let f_1_uv = self
            .nonzero_gates
            .par_iter()
            .fold(
                || F::ZERO,
                |acc, (z_ind, x_ind)| {
                    let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                        (
                            beta_g1.f.get(*z_ind as usize).unwrap_or(F::ZERO),
                            beta_u.f.get(*x_ind as usize).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                &claim_challenges[self.num_dataparallel_vars..],
                                *z_ind as usize,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                &round_challenges[self.num_dataparallel_vars..],
                                *x_ind as usize,
                            ),
                        )
                    };

                    acc + gz * ux
                },
            )
            .sum::<F>();

        #[cfg(not(feature = "parallel"))]
        let f_1_uv = self
            .nonzero_gates
            .iter()
            .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                let (gz, ux) = if let Some((beta_u, beta_g1)) = &beta_ug {
                    (
                        beta_g1.f.get(*z_ind as usize).unwrap_or(F::ZERO),
                        beta_u.f.get(*x_ind as usize).unwrap_or(F::ZERO),
                    )
                } else {
                    (
                        BetaValues::compute_beta_over_challenge_and_index(
                            &claim_challenges[self.num_dataparallel_vars..],
                            *z_ind as usize,
                        ),
                        BetaValues::compute_beta_over_challenge_and_index(
                            &round_challenges[self.num_dataparallel_vars..],
                            *x_ind as usize,
                        ),
                    )
                };

                acc + gz * ux
            });

        let beta_bound = if self.num_dataparallel_vars != 0 {
            let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &round_challenges[..self.num_dataparallel_vars],
            )
        } else {
            F::ONE
        };

        PostSumcheckLayer(vec![Product::<F, F>::new(
            &[self.source_mle.clone()],
            f_1_uv * beta_bound,
        )])
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
        let mut claims = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];

        for index in self.source_mle.mle_indices() {
            fixed_mle_indices_u.push(
                index
                    .val()
                    .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
            );
        }
        let val = self.source_mle.first();
        let claim: Claim<F> = Claim::new(
            fixed_mle_indices_u,
            val,
            self.layer_id(),
            self.source_mle.layer_id(),
        );
        claims.push(claim);

        Ok(claims)
    }
}

/// identity gate struct
/// wiring is used to be able to select specific elements from an MLE
/// (equivalent to an add gate with a zero mle ref)
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct IdentityGate<F: Field> {
    /// layer id that this gate is found
    pub layer_id: LayerId,
    /// we only need a single incoming gate and a single outgoing gate so this is a
    /// tuple of 2 integers representing which label maps to which
    /// Tuples are of form `(dest_idx, src_idx)`.
    pub nonzero_gates: Vec<(u32, u32)>,
    /// the mle ref in question from which we are selecting specific indices
    pub source_mle: DenseMle<F>,
    /// the beta table which enumerates the incoming claim's challenge points on the MLE
    beta_g1: Option<MultilinearExtension<F>>,
    /// The [BetaValues] struct which enumerates the incoming claim's challenge points on the
    /// dataparallel vars of the MLE
    beta_g2: Option<BetaValues<F>>,
    /// The MLE initialized in the dataparallel phase which contains the nonzero gate
    /// evaluations folded into a size 2^(num_dataparallel_bits) bookkeeping table.
    dataparallel_af2_mle: Option<DenseMle<F>>,
    /// The challenges pertaining to the `x` variables, which are the non-dataparallel
    /// variables in the source MLE.
    g1_challenges: Option<Vec<F>>,
    /// The challenges pertaining to the dataparallel variables in the source MLE.
    g2_challenges: Option<Vec<F>>,
    /// The MLE initialized in phase 1, which contains the beta values over `g1_challenges`
    /// folded into the wiring function.
    a_hg_mle_phase_1: Option<DenseMle<F>>,
    /// The total number of variables in the layer.
    pub total_num_vars: usize,
    /// The number of vars representing the number of "dataparallel" copies of the circuit.
    pub num_dataparallel_vars: usize,
}

impl<F: Field> IdentityGate<F> {
    /// Create a new [IdentityGate] struct.
    pub fn new(
        layer_id: LayerId,
        nonzero_gates: Vec<(u32, u32)>,
        mle: DenseMle<F>,
        total_num_vars: usize,
        num_dataparallel_vars: usize,
    ) -> IdentityGate<F> {
        IdentityGate {
            layer_id,
            nonzero_gates,
            source_mle: mle,
            beta_g1: None,
            beta_g2: None,
            dataparallel_af2_mle: None,
            a_hg_mle_phase_1: None,
            total_num_vars,
            num_dataparallel_vars,
            g1_challenges: None,
            g2_challenges: None,
        }
    }

    fn set_beta_g1(&mut self, beta_g1: MultilinearExtension<F>) {
        self.beta_g1 = Some(beta_g1);
    }

    fn set_beta_g2(&mut self, beta_g2: BetaValues<F>) {
        self.beta_g2 = Some(beta_g2);
    }

    fn set_g1_challenges(&mut self, g1: Vec<F>) {
        self.g1_challenges = Some(g1);
    }

    fn set_g2_challenges(&mut self, g2: Vec<F>) {
        self.g2_challenges = Some(g2);
    }

    fn append_leaf_mles_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<F>) {
        assert_eq!(self.source_mle.len(), 1);
        transcript_writer.append("Fully bound source MLE", self.source_mle.first());
    }

    fn init_dataparallel_phase(&mut self, g1_challenges: Vec<F>) {
        let beta_getter: Box<dyn Fn(usize) -> F> = if !global_prover_lazy_beta_evals() {
            let beta_g1 = BetaValues::new_beta_equality_mle(g1_challenges);
            Box::new(move |idx| beta_g1.get(idx).unwrap_or(F::ZERO))
        } else {
            Box::new(|idx| BetaValues::compute_beta_over_challenge_and_index(&g1_challenges, idx))
        };
        let num_dataparallel_copies = 1 << self.num_dataparallel_vars;
        let num_nondataparallel_coeffs =
            1 << (self.source_mle.num_free_vars() - self.num_dataparallel_vars);
        let mut a_f2 = vec![F::ZERO; 1 << (self.num_dataparallel_vars)];
        (0..num_dataparallel_copies).for_each(|idx| {
            let mut adder_f2 = F::ZERO;
            self.nonzero_gates.iter().for_each(|(z, x)| {
                let gz = beta_getter(*z as usize);
                let f2_val = self
                    .source_mle
                    .mle
                    .get((*x as usize) + (idx * num_nondataparallel_coeffs))
                    .unwrap_or(F::ZERO);

                adder_f2 += gz * f2_val;
            });
            a_f2[idx] += adder_f2;
        });

        let mut af2_mle = DenseMle::new_from_raw(a_f2, self.layer_id());
        af2_mle.index_mle_indices(0);
        self.dataparallel_af2_mle = Some(af2_mle);
    }

    /// initialize necessary bookkeeping tables by traversing the nonzero gates
    pub fn init_phase_1(&mut self, challenge: Vec<F>) {
        if !global_prover_lazy_beta_evals() {
            let beta_g1 = BetaValues::new_beta_equality_mle(challenge.clone());
            self.set_beta_g1(beta_g1);
        }

        let num_vars = self.source_mle.num_free_vars();

        let mut a_hg_mle_vec = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates.iter().for_each(|(z_ind, x_ind)| {
            let beta_g_at_z = if global_prover_lazy_beta_evals() {
                BetaValues::compute_beta_over_challenge_and_index(&challenge, *z_ind as usize)
            } else {
                self.beta_g1
                    .as_ref()
                    .unwrap()
                    .get(*z_ind as usize)
                    .unwrap_or(F::ZERO)
            };

            a_hg_mle_vec[*x_ind as usize] += beta_g_at_z;
        });

        let mut a_hg_mle = DenseMle::new_from_raw(a_hg_mle_vec, self.layer_id());
        a_hg_mle.index_mle_indices(self.num_dataparallel_vars);

        self.a_hg_mle_phase_1 = Some(a_hg_mle);
    }

    /// Get the evals for an identity gate. Note that this specifically
    /// refers to computing the prover message while binding the dataparallel bits of a `Gate`
    /// expression.
    fn compute_sumcheck_message_data_parallel_identity_gate_beta_cascade(
        &self,
        round_index: usize,
    ) -> Result<Vec<F>> {
        // When we have an identity gate, we have to multiply the beta table over the dataparallel challenges
        // with the function on the x variables.
        let degree = 2;
        let a_f2_x = self.dataparallel_af2_mle.as_ref().unwrap();
        let beta_values = self.beta_g2.as_ref().unwrap();
        let (unbound_beta_values, bound_beta_values) =
            beta_values.get_relevant_beta_unbound_and_bound(a_f2_x.mle_indices());
        let evals = beta_cascade(
            &[a_f2_x],
            degree,
            round_index,
            &unbound_beta_values,
            &bound_beta_values,
        )
        .0;

        Ok(evals)
    }
}

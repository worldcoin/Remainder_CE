//! Identity gate id(z, x) determines whether the xth gate from the
//! i + 1th layer contributes to the zth gate in the ith layer.

use std::collections::HashSet;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use tracing_subscriber::layer::Identity;

use crate::{
    claims::{Claim, ClaimError, RawClaim},
    layer::{gate::gate_helpers::bind_round_identity, LayerError, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, verifier_mle::VerifierMle, Mle, MleIndex,
    },
    sumcheck::*,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};

use crate::layer::gate::gate_helpers::compute_sumcheck_message_identity;

use thiserror::Error;

use super::{
    gate::{
        gate_helpers::{
            compute_sumcheck_messages_data_parallel_identity_gate,
            evaluate_mle_ref_product_no_beta_table, prove_round_identity_gate_dataparallel_phase,
        },
        index_mle_indices_gate, GateError,
    },
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    Layer, LayerDescription, LayerId, VerifierLayer,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Controls whether the `beta` optimiation should be enabled. When enabled, all
/// functions in this module that compute the value of a `beta` function at a
/// given index, will compute its value lazily using
/// [BetaValues::compute_beta_over_challenge_and_index] instead of pre-computing
/// and storing the entire bookkeeping table.
const LAZY_BETA_EVALUATION: bool = true;

/// The circuit Description for an [IdentityGate].
#[derive(Serialize, Deserialize, Debug, Clone, Hash)]
#[serde(bound = "F: Field")]
pub struct IdentityGateLayerDescription<F: Field> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is
    /// the output of adding all values from labels x.
    wiring: Vec<(usize, usize)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: MleDescription<F>,

    /// The number of vars representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_vars: usize,
}

impl<F: Field> Display for IdentityGateLayerDescription<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: IdentityGateLayer with #wirings={}, re-routing values from layer {}", self.id, self.wiring.len(), self.source_mle.layer_id())
    }
}

impl<F: Field> IdentityGateLayerDescription<F> {
    /// Constructor for the [IdentityGateLayerDescription] using the gate wiring, the source mle
    /// for the rerouting, and the layer_id.
    pub fn new(
        id: LayerId,
        wiring: Vec<(usize, usize)>,
        source_mle: MleDescription<F>,
        num_dataparallel_vars: Option<usize>,
    ) -> Self {
        Self {
            id,
            wiring,
            source_mle,
            num_dataparallel_vars: num_dataparallel_vars.unwrap_or(0),
        }
    }
}

/// Degree of independent variable is always quadratic!
/// (regardless of if there's dataparallel or not)
/// V_i(g_2, g_1) = \sum_{p_2} \sum_{x} \beta(g_2, p_2) f_1(g_1, x) (V_{i + 1}(p_2, x))
const ID_NUM_EVALS: usize = 3;

impl<F: Field> LayerDescription<F> for IdentityGateLayerDescription<F> {
    type VerifierLayer = VerifierIdentityGateLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.id
    }

    /// Note that this ONLY verifies for non-dataparallel identity gate!!!
    ///
    /// TODO(vishady, ryancao): Implement dataparallel identity gate prover + verifier
    fn verify_rounds(
        &self,
        claim: RawClaim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        let _num_sumcheck_rounds = self.sumcheck_round_indices().len();

        // --- Store challenges for later claim generation ---
        let mut challenges = vec![];

        // --- WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL ---
        // --- INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED ---
        // --- vars ---
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

        // --- Grab the first round prover sumcheck message g_1(x) ---
        let mut sumcheck_messages: Vec<Vec<F>> = vec![];
        let first_round_sumcheck_messages =
            transcript_reader.consume_elements("Initial sumcheck evaluations", ID_NUM_EVALS)?;
        sumcheck_messages.push(first_round_sumcheck_messages.clone());

        // Check: V_i(g_1) =? g_1(0) + g_1(1)
        // TODO(ryancao): SUPER overloaded notation (in e.g. above comments); fix across the board
        if first_round_sumcheck_messages[0] + first_round_sumcheck_messages[1] != claim.get_eval() {
            return Err(VerificationError::SumcheckStartFailed);
        }

        for _sumcheck_round_idx in 1..self.num_dataparallel_vars + num_u {
            // --- Read challenge r_{i - 1} from transcript ---
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
                .unwrap();
            let g_i_minus_1_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();

            // --- Evaluate g_{i - 1}(r_{i - 1}) ---
            let prev_at_r = evaluate_at_a_point(&g_i_minus_1_evals, challenge).unwrap();

            // --- Read off g_i(0), g_i(1), ..., g_i(d) from transcript ---
            let curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", ID_NUM_EVALS)
                .unwrap();

            // --- Check: g_i(0) + g_i(1) =? g_{i - 1}(r_{i - 1}) ---
            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(VerificationError::SumcheckFailed);
            };

            // --- Add the prover message to the sumcheck messages ---
            sumcheck_messages.push(curr_evals);

            // --- Store all challenges from transcript ---
            challenges.push(challenge);
        }

        // final round of sumcheck
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        let verifier_id_gate_layer = self
            .convert_into_verifier_layer(&challenges, claim.get_point(), transcript_reader)
            .unwrap();
        let final_result = verifier_id_gate_layer.evaluate(&claim);

        // Finally, compute g_n(r_n).
        let g_n_evals = sumcheck_messages[sumcheck_messages.len() - 1].clone();
        let prev_at_r = evaluate_at_a_point(&g_n_evals, final_chal).unwrap();

        // error if this doesn't match the last round of sumcheck
        if final_result != prev_at_r {
            return Err(VerificationError::FinalSumcheckFailed);
        }

        Ok(VerifierLayerEnum::IdentityGate(verifier_id_gate_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
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

        (0..num_u + self.num_dataparallel_vars).collect_vec()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_challenges: &[F],
        _claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        // --- WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL ---
        // --- INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR FREE/INDEXED ---
        // --- vars ---
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

        // --- Create the resulting verifier layer for claim tracking ---
        // TODO(ryancao): This is not necessary; we only need to pass back the actual claims
        let verifier_id_gate_layer = VerifierIdentityGateLayer {
            layer_id: self.layer_id(),
            wiring: self.wiring.clone(),
            source_mle: src_verifier_mle,
            first_u_challenges,
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
        let beta_ug = if !LAZY_BETA_EVALUATION {
            Some((
                BetaValues::new_beta_equality_mle(round_challenges.to_vec()),
                BetaValues::new_beta_equality_mle(claim_challenges.to_vec()),
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
                    let (gz, ux) = if let Some((beta_u, beta_g)) = &beta_ug {
                        (
                            beta_g.mle.f.get(*z_ind).unwrap_or(F::ZERO),
                            beta_u.mle.f.get(*x_ind).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                claim_challenges,
                                *z_ind,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                round_challenges,
                                *x_ind,
                            ),
                        )
                    };

                    acc + gz * ux
                },
            )
            .sum::<F>();

        #[cfg(not(feature = "parallel"))]
        let f_1_uv = self.wiring.iter().fold(F::ZERO, |acc, (z_ind, x_ind)| {
            let (gz, ux) = if let Some((beta_u, beta_g)) = &beta_ug {
                (
                    beta_g.mle.f.get(*z_ind).unwrap_or(F::ZERO),
                    beta_u.mle.f.get(*x_ind).unwrap_or(F::ZERO),
                )
            } else {
                (
                    BetaValues::compute_beta_over_challenge_and_index(claim_challenges, *z_ind),
                    BetaValues::compute_beta_over_challenge_and_index(round_challenges, *x_ind),
                )
            };

            acc + gz * ux
        });

        PostSumcheckLayer(vec![Product::<F, Option<F>>::new(
            &[self.source_mle.clone()],
            f_1_uv,
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
        let num_dataparallel_vars = if self.num_dataparallel_vars == 0 {
            None
        } else {
            Some(self.num_dataparallel_vars)
        };
        let id_gate_layer = IdentityGate::new(
            self.layer_id(),
            self.wiring.clone(),
            source_mle,
            num_dataparallel_vars,
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

        let max_gate_val = self
            .wiring
            .iter()
            .fold(&0, |acc, (z, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel vars, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel vars
        let num_dataparallel_vals = 1 << (self.num_dataparallel_vars);
        let res_table_num_entries =
            ((max_gate_val + 1) * num_dataparallel_vals).next_power_of_two();

        let mut remap_table = vec![F::ZERO; res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            self.wiring.iter().for_each(|(z, x)| {
                let id_val = source_mle_data
                    .f
                    .get(idx + (x * num_dataparallel_vals))
                    .unwrap_or(F::ZERO);
                remap_table[idx + z * num_dataparallel_vals] = id_val;
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

        let beta_ug = if LAZY_BETA_EVALUATION {
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
                    let (gz, ux) = if let Some((beta_u, beta_g)) = &beta_ug {
                        (
                            beta_g.mle.f.get(*z_ind).unwrap_or(F::ZERO),
                            beta_u.mle.f.get(*x_ind).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                &g1_challenges,
                                *z_ind,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                &self.first_u_challenges,
                                *x_ind,
                            ),
                        )
                    };

                    acc + gz * ux
                },
            )
            .sum::<F>();

        #[cfg(not(feature = "parallel"))]
        let f_1_uv = self.wiring.iter().fold(F::ZERO, |acc, (z_ind, x_ind)| {
            let (gz, ux) = if let Some((beta_u, beta_g)) = &beta_ug {
                (
                    beta_g.mle.f.get(*z_ind).unwrap_or(F::ZERO),
                    beta_u.mle.f.get(*x_ind).unwrap_or(F::ZERO),
                )
            } else {
                (
                    BetaValues::compute_beta_over_challenge_and_index(&g1_challenges, *z_ind),
                    BetaValues::compute_beta_over_challenge_and_index(
                        &self.first_u_challenges,
                        *x_ind,
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
    wiring: Vec<(usize, usize)>,

    /// The source MLE of the expression, i.e. the mle that makes up the "x"
    /// variables.
    source_mle: VerifierMle<F>,

    /// The challenges for `x`, as derived from sumcheck.
    first_u_challenges: Vec<F>,

    /// The number of dataparallel rounds.
    num_dataparallel_rounds: usize,

    /// The challenges for `p_2`, as derived from sumcheck.
    dataparallel_sumcheck_challenges: Vec<F>,
}

impl<F: Field> VerifierLayer<F> for VerifierIdentityGateLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        // Grab the claim on the left side.
        // TODO!(ryancao): Do error handling here!
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

/// implement the layer trait for identitygate struct
impl<F: Field> Layer<F> for IdentityGate<F> {
    fn prove(
        &mut self,
        claim: RawClaim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        let (mut beta_g1, mut beta_g2) = self.compute_beta_tables(claim.get_point());
        let mut beta_g2_fully_bound = F::ONE;

        // We perform the dataparallel initialization only if there is at least one variable
        // representing which copy we are in.
        if self.num_dataparallel_vars > 0 {
            beta_g2_fully_bound = self
                .perform_dataparallel_phase(&mut beta_g1, &mut beta_g2, transcript_writer)
                .unwrap();
        }

        // initialization, get the first sumcheck message
        let first_message = self
            .init_phase_1(claim.get_point()[self.num_dataparallel_vars..].to_vec())
            .expect("could not evaluate original lhs and rhs")
            .into_iter()
            .map(|eval| eval * beta_g2_fully_bound)
            .collect_vec();

        let phase_1_mle_refs = self
            .phase_1_mles
            .as_mut()
            .ok_or(GateError::Phase1InitError)
            .unwrap();

        let mut challenges: Vec<F> = vec![];
        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);
        let num_rounds = self.mle_ref.num_free_vars();

        // sumcheck rounds (binding x)
        let _sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                // if there are copy vars, we want to start at that index
                bind_round_identity(
                    round + self.num_dataparallel_vars,
                    challenge,
                    phase_1_mle_refs,
                );
                let phase_1_mle_references: Vec<&DenseMle<F>> = phase_1_mle_refs.iter().collect();
                let eval = compute_sumcheck_message_identity(
                    round + self.num_dataparallel_vars,
                    &phase_1_mle_references,
                )
                .unwrap()
                .into_iter()
                .map(|eval| eval * beta_g2_fully_bound)
                .collect_vec();
                transcript_writer.append_elements("Sumcheck evaluations", &eval);
                Ok::<_, LayerError>(eval)
            }))
            .try_collect()?;

        // final challenge after binding x (left side of the sum)
        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
        challenges.push(final_chal);

        phase_1_mle_refs.iter_mut().for_each(|mle| {
            mle.fix_variable(num_rounds - 1 + self.num_dataparallel_vars, final_chal);
        });

        // --- Finally, send the claimed values for each of the bound MLE to the verifier ---
        // First, send the claimed value of V_{i + 1}(u)
        let source_mle_reduced = self.phase_1_mles.clone().unwrap()[1].clone();
        debug_assert!(source_mle_reduced.len() == 1);
        transcript_writer.append(
            "Evaluation of V_{i + 1}(g_2, u)",
            source_mle_reduced.first(),
        );
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    // TODO!(ende): no references in codebase as of now, if so, add data parallel support
    fn initialize(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        if !LAZY_BETA_EVALUATION {
            let beta_g = BetaValues::new_beta_equality_mle(claim_point.to_vec());
            self.set_beta_g(beta_g);
        }

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_free_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = if LAZY_BETA_EVALUATION {
                    BetaValues::compute_beta_over_challenge_and_index(claim_point, z_ind)
                } else {
                    self.beta_g
                        .as_ref()
                        .unwrap()
                        .mle
                        .get(z_ind)
                        .unwrap_or(F::ZERO)
                };
                a_hg_mle_ref[x_ind] += beta_g_at_z;
            });

        let mut phase_1 = [
            DenseMle::new_from_raw(a_hg_mle_ref, LayerId::Input(0)),
            self.mle_ref.clone(),
        ];

        index_mle_indices_gate(&mut phase_1, 0);
        self.set_phase_1(phase_1.clone());

        Ok(())
    }

    // TODO!(ende): no references in codebase as of now, if so, add data parallel support
    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError> {
        let mles: Vec<&DenseMle<F>> = self.phase_1_mles.as_ref().unwrap().iter().collect();
        let independent_variable = mles
            .iter()
            .map(|mle_ref| {
                mle_ref
                    .mle_indices()
                    .contains(&MleIndex::Indexed(round_index))
            })
            .reduce(|acc, item| acc | item)
            .unwrap();
        let evals = evaluate_mle_ref_product_no_beta_table(&mles, independent_variable, mles.len())
            .unwrap();
        let SumcheckEvals(evaluations) = evals;
        Ok(evaluations)
    }

    // TODO!(ende): no references in codebase as of now
    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        let mles = self.phase_1_mles.as_mut().unwrap();
        mles.iter_mut().for_each(|mle_ref| {
            mle_ref.fix_variable(round_index, challenge);
        });
        Ok(())
    }

    // TODO!(ende): no references in codebase as of now
    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.mle_ref.num_free_vars()).collect_vec()
    }

    // TODO!(ende): no references in codebase as of now
    fn max_degree(&self) -> usize {
        2
    }

    // TODO!(ende): no references in codebase as of now
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let [_, mle_ref] = self.phase_1_mles.as_ref().unwrap();
        let beta_u = if !LAZY_BETA_EVALUATION {
            Some(BetaValues::new_beta_equality_mle(round_challenges.to_vec()))
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
                    let (gz, ux) = if let Some(beta_u) = &beta_u {
                        (
                            self.beta_g
                                .as_ref()
                                .unwrap()
                                .mle
                                .get(*z_ind)
                                .unwrap_or(F::ZERO),
                            beta_u.mle.get(*x_ind).unwrap_or(F::ZERO),
                        )
                    } else {
                        (
                            BetaValues::compute_beta_over_challenge_and_index(
                                claim_challenges,
                                *z_ind,
                            ),
                            BetaValues::compute_beta_over_challenge_and_index(
                                round_challenges,
                                *x_ind,
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
                let (gz, ux) = if let Some(beta_u) = &beta_u {
                    (
                        self.beta_g
                            .as_ref()
                            .unwrap()
                            .mle
                            .get(*z_ind)
                            .unwrap_or(F::ZERO),
                        beta_u.mle.get(*x_ind).unwrap_or(F::ZERO),
                    )
                } else {
                    (
                        BetaValues::compute_beta_over_challenge_and_index(claim_challenges, *z_ind),
                        BetaValues::compute_beta_over_challenge_and_index(round_challenges, *x_ind),
                    )
                };

                acc + gz * ux
            });

        PostSumcheckLayer(vec![Product::<F, F>::new(&[mle_ref.clone()], f_1_uv)])
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
        let mut claims = vec![];
        let mut fixed_mle_indices_u: Vec<F> = vec![];

        // check the left side of the sum (f2(u)) against the challenges made to bind that variable
        if let Some([_, mle_ref]) = &self.phase_1_mles {
            for index in mle_ref.mle_indices() {
                fixed_mle_indices_u.push(
                    index
                        .val()
                        .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))?,
                );
            }
            let val = mle_ref.first();
            let claim: Claim<F> = Claim::new(
                fixed_mle_indices_u,
                val,
                self.layer_id(),
                mle_ref.layer_id(),
            );
            claims.push(claim);
        } else {
            return Err(LayerError::LayerNotReady);
        }

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
    pub nonzero_gates: Vec<(usize, usize)>,
    /// the mle ref in question from which we are selecting specific indices
    pub mle_ref: DenseMle<F>,
    /// the beta table which enumerates the incoming claim's challenge points
    beta_g: Option<DenseMle<F>>,
    /// the mles that are created from the initial phase, where we automatically
    /// filter through the nonzero gates using the libra trick
    pub phase_1_mles: Option<[DenseMle<F>; 2]>,
    /// The number of vars representing the number of "dataparallel" copies of the circuit.
    pub num_dataparallel_vars: usize,
}

impl<F: Field> IdentityGate<F> {
    /// new addgate mle (wrapper constructor)
    pub fn new(
        layer_id: LayerId,
        nonzero_gates: Vec<(usize, usize)>,
        mle_ref: DenseMle<F>,
        num_dataparallel_vars: Option<usize>,
    ) -> IdentityGate<F> {
        IdentityGate {
            layer_id,
            nonzero_gates,
            mle_ref,
            beta_g: None,
            phase_1_mles: None,
            num_dataparallel_vars: num_dataparallel_vars.unwrap_or(0),
        }
    }

    fn compute_beta_tables(&mut self, challenges: &[F]) -> (DenseMle<F>, DenseMle<F>) {
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
        let mut beta_g2 = BetaValues::new_beta_equality_mle(g2_challenges);
        beta_g2.index_mle_indices(0);
        let beta_g1 = BetaValues::new_beta_equality_mle(g1_challenges);

        (beta_g1, beta_g2)
    }

    /// Initialize the dataparallel phase: construct the necessary mles and return the first sumcheck message.
    /// This will then set the necessary fields of the [Gate] struct so that the dataparallel vars can be
    /// correctly bound during the first `num_dataparallel_vars` rounds of sumcheck.
    fn init_dataparallel_phase(
        &mut self,
        beta_g1: &mut DenseMle<F>,
        beta_g2: &mut DenseMle<F>,
    ) -> Result<Vec<F>, GateError> {
        // Index original bookkeeping tables.
        self.mle_ref.index_mle_indices(0);

        // Result of initializing is the first sumcheck message.

        compute_sumcheck_messages_data_parallel_identity_gate(
            &self.mle_ref,
            beta_g2,
            beta_g1,
            &self.nonzero_gates,
            self.num_dataparallel_vars,
        )
    }

    // Once the initialization of the dataparallel phase is done, we can perform the dataparallel phase.
    // This means that we are binding all vars that represent which copy of the circuit we are in.
    fn perform_dataparallel_phase(
        &mut self,
        beta_g1: &mut DenseMle<F>,
        beta_g2: &mut DenseMle<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<F, LayerError> {
        // Initialization, first message comes from here.
        let mut challenges: Vec<F> = vec![];

        let first_message = self.init_dataparallel_phase(beta_g1, beta_g2).expect(
            "could not evaluate original lhs and rhs in order to get first sumcheck message",
        );

        let mle_ref = &mut self.mle_ref;

        transcript_writer
            .append_elements("Initial Sumcheck evaluations DATAPARALLEL", &first_message);
        let num_rounds_copy_phase = self.num_dataparallel_vars;

        // Do the first dataparallel vars number sumcheck rounds using libra giraffe.
        let _sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_rounds_copy_phase).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge DATAPARALLEL");
                challenges.push(challenge);
                let eval = prove_round_identity_gate_dataparallel_phase(
                    mle_ref,
                    beta_g1,
                    beta_g2,
                    round,
                    challenge,
                    &self.nonzero_gates,
                    self.num_dataparallel_vars - round,
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
        self.mle_ref
            .fix_variable(num_rounds_copy_phase - 1, final_chal_copy);

        if beta_g2.len() == 1 {
            let beta_g2_fully_bound = beta_g2.first();
            Ok(beta_g2_fully_bound)
        } else {
            Err(LayerError::LayerNotReady)
        }
    }

    fn set_beta_g(&mut self, beta_g: DenseMle<F>) {
        self.beta_g = Some(beta_g);
    }

    /// bookkeeping tables necessary for binding x
    fn set_phase_1(&mut self, mle_refs: [DenseMle<F>; 2]) {
        self.phase_1_mles = Some(mle_refs);
    }

    /// initialize necessary bookkeeping tables by traversing the nonzero gates
    pub fn init_phase_1(&mut self, challenge: Vec<F>) -> Result<Vec<F>, GateError> {
        if !LAZY_BETA_EVALUATION {
            let beta_g = BetaValues::new_beta_equality_mle(challenge.clone());
            self.set_beta_g(beta_g);
        }

        self.mle_ref.index_mle_indices(0);
        let num_vars = self.mle_ref.num_free_vars();

        let mut a_hg_mle_ref = vec![F::ZERO; 1 << num_vars];

        self.nonzero_gates
            .clone()
            .into_iter()
            .for_each(|(z_ind, x_ind)| {
                let beta_g_at_z = if LAZY_BETA_EVALUATION {
                    BetaValues::compute_beta_over_challenge_and_index(&challenge, z_ind)
                } else {
                    self.beta_g
                        .as_ref()
                        .unwrap()
                        .mle
                        .get(z_ind)
                        .unwrap_or(F::ZERO)
                };

                a_hg_mle_ref[x_ind] += beta_g_at_z;
            });

        let mut phase_1 = [
            DenseMle::new_from_raw(a_hg_mle_ref, LayerId::Input(0)),
            self.mle_ref.clone(),
        ];

        index_mle_indices_gate(&mut phase_1, self.num_dataparallel_vars);
        self.set_phase_1(phase_1.clone());

        let independent_variable = phase_1
            .iter()
            .map(|mle_ref| {
                mle_ref
                    .mle_indices()
                    .contains(&MleIndex::Indexed(self.num_dataparallel_vars))
            })
            .reduce(|acc, item| acc | item)
            .ok_or(GateError::EmptyMleList)?;
        let phase_1_mle_references: Vec<&DenseMle<F>> = phase_1.iter().collect();
        let evals = evaluate_mle_ref_product_no_beta_table(
            &phase_1_mle_references,
            independent_variable,
            phase_1.len(),
        )
        .unwrap();

        let SumcheckEvals(evaluations) = evals;
        Ok(evaluations)
    }
}

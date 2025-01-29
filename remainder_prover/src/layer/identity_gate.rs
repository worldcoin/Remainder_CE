//! Identity gate id(z, x) determines whether the xth gate from the i + 1th
//! layer contributes to the zth gate in the ith layer.

use std::{
    cmp::Ordering,
    collections::HashSet,
    fmt::{Debug, Formatter},
};

use crate::{
    claims::{Claim, ClaimError, RawClaim},
    layer::{
        gate::gate_helpers::compute_fully_bound_identity_gate_function, LayerError,
        VerificationError,
    },
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{
        betavalues::BetaValues, dense::DenseMle, evals::MultilinearExtension,
        mle_description::MleDescription, verifier_mle::VerifierMle, Mle, MleIndex,
    },
    sumcheck::*,
};
use itertools::Itertools;
use remainder_shared_types::{
    config::{global_config::global_claim_agg_strategy, ClaimAggregationStrategy},
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use thiserror::Error;

use super::{
    gate::gate_helpers::{
        compute_sumcheck_message_data_parallel_identity_gate, evaluate_mle_product_no_beta_table,
        fold_wiring_into_beta_mle_identity_gate,
    },
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    Layer, LayerDescription, LayerId, VerifierLayer,
};

/// The circuit Description for an [IdentityGate].
#[derive(Serialize, Deserialize, Clone, Hash)]
#[serde(bound = "F: Field")]
pub struct IdentityGateLayerDescription<F: Field> {
    /// The layer id associated with this gate layer.
    id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is the
    /// output of adding all values from labels x.
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
    /// Constructor for [IdentityGateLayerDescription]. Arguments:
    /// * `id`: The layer id associated with this layer.
    /// * `source_mle`: The Mle that is being routed to this layer.
    /// * `nonzero_gates`: A list of tuples representing the gates that are
    ///   nonzero, in the form `(dest_idx, src_idx)`.
    /// * `total_num_vars`: The total number of variables in the layer.
    /// * `num_dataparallel_vars`: The number of dataparallel variables to use
    ///   in this layer.
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
        claims: &[&RawClaim<F>],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // Random coefficients depending on claim aggregation strategy.
        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                assert_eq!(claims.len(), 1);
                vec![F::ONE]
            }
            ClaimAggregationStrategy::RLC => {
                transcript_reader.get_challenges("RLC Claim Agg Coefficients", claims.len())?
            }
        };

        // Represents `g_{i-1}(x)` of the previous round. This is initialized to
        // the constant polynomial `g_0(x)` which evaluates to the claim result
        // for any `x`.
        let mut g_prev_round = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                vec![claims[0].get_eval()]
            }
            ClaimAggregationStrategy::RLC => vec![random_coefficients
                .iter()
                .zip(claims)
                .fold(F::ZERO, |acc, (rlc_val, claim)| {
                    acc + *rlc_val * claim.get_eval()
                })],
        };

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = F::ZERO;

        let num_rounds = self.sumcheck_round_indices().len();

        // For round 1 <= i <= n, perform the check:
        for _round in 0..num_rounds {
            // Degree of independent variable is always quadratic! (regardless
            // of if there's dataparallel or not) V_i(g_2, g_1) = \sum_{p_2}
            // \sum_{x} \beta(g_2, p_2) f_1(g_1, x) (V_{i + 1}(p_2, x))
            let degree = 2;

            let g_cur_round = transcript_reader
                .consume_elements("Sumcheck message", degree + 1)
                .map_err(VerificationError::TranscriptError)?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_challenge("Sumcheck challenge")?;

            // Verify that: `g_i(0) + g_i(1) == g_{i - 1}(r_{i-1})`
            let g_i_zero = evaluate_at_a_point(&g_cur_round, F::ZERO).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, F::ONE).unwrap();
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();

            if g_i_zero + g_i_one != g_prev_r_prev {
                dbg!(_round);
                return Err(VerificationError::SumcheckFailed);
            }

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // Evalute `g_n(r_n)`. Note: If there were no nonlinear rounds, this
        // value reduces to `claim.get_result()` due to how we initialized
        // `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        let verifier_id_gate_layer = self
            .convert_into_verifier_layer(
                &challenges,
                &claims.iter().map(|claim| claim.get_point()).collect_vec(),
                transcript_reader,
            )
            .unwrap();
        let final_result = verifier_id_gate_layer.evaluate(
            &claims.iter().map(|claim| claim.get_point()).collect_vec(),
            &random_coefficients,
        );

        if g_final_r_final != final_result {
            return Err(VerificationError::FinalSumcheckFailed);
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
        _claim_points: &[&[F]],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        // WARNING: WE ARE ASSUMING HERE THAT MLE INDICES INCLUDE DATAPARALLEL
        // INDICES AND MAKE NO DISTINCTION BETWEEN THOSE AND REGULAR
        // FREE/INDEXED vars
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

        // We want to separate the challenges into which ones are from the
        // dataparallel vars, which ones and are for binding x (phase 1)
        let mut sumcheck_bindings_vec = sumcheck_challenges.to_vec();
        let first_u_challenges = sumcheck_bindings_vec.split_off(self.num_dataparallel_vars);
        let dataparallel_sumcheck_challenges = sumcheck_bindings_vec;

        assert_eq!(first_u_challenges.len(), num_u);

        // Since the original mles are dataparallel, the challenges are the
        // concat of the copy vars and the variable bound vars.
        let src_verifier_mle = self
            .source_mle
            .into_verifier_mle(sumcheck_challenges, transcript_reader)
            .unwrap();

        // Create the resulting verifier layer for claim tracking TODO(ryancao):
        // This is not necessary; we only need to pass back the actual claims
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
        let beta_bound = if self.num_dataparallel_vars != 0 {
            let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &round_challenges[..self.num_dataparallel_vars],
            )
        } else {
            F::ONE
        };

        let f_1_gu = compute_fully_bound_identity_gate_function(
            &round_challenges[self.num_dataparallel_vars..],
            &[&claim_challenges[self.num_dataparallel_vars..]],
            &self.wiring,
            &[beta_bound],
        );

        PostSumcheckLayer(vec![Product::<F, Option<F>>::new(
            &[self.source_mle.clone()],
            f_1_gu,
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
    /// Computes the oracle query's value for a given
    /// [IdentityGateVerifierLayer].
    pub fn evaluate(&self, claim_points: &[&[F]], random_coefficients: &[F]) -> F {
        assert_eq!(random_coefficients.len(), claim_points.len());
        let scaled_random_coeffs = claim_points
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

        let f_1_gu = compute_fully_bound_identity_gate_function(
            &self.first_u_challenges,
            &claim_points
                .iter()
                .map(|claim| &claim[self.num_dataparallel_rounds..])
                .collect_vec(),
            &self.wiring,
            &scaled_random_coeffs,
        );
        // get the fully evaluated "expression"
        f_1_gu * self.source_mle.value()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
/// The layer representing a fully bound [IdentityGate].
pub struct VerifierIdentityGateLayer<F: Field> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// A vector of tuples representing the "nonzero" gates, especially useful
    /// in the sparse case the format is (z, x) where the gate at label z is the
    /// output of adding all values from labels x.
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

    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
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
        claims: &[&RawClaim<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                assert_eq!(claims.len(), 1);
                self.initialize(claims[0].get_point())?;
                vec![F::ONE]
            }
            ClaimAggregationStrategy::RLC => {
                let random_coefficients =
                    transcript_writer.get_challenges("RLC Claim Agg Coefficients", claims.len());
                self.initialize_rlc(&random_coefficients, claims);
                random_coefficients
            }
        };
        let sumcheck_indices = self.sumcheck_round_indices();
        (sumcheck_indices.iter()).for_each(|round_idx| {
            let sumcheck_message = self
                .compute_round_sumcheck_message(*round_idx, &random_coefficients)
                .unwrap();
            transcript_writer.append_elements("Round sumcheck message", &sumcheck_message);
            let challenge = transcript_writer.get_challenge("Sumcheck challenge");
            self.bind_round_variable(*round_idx, challenge).unwrap();
        });
        self.append_leaf_mles_to_transcript(transcript_writer);
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn initialize(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        self.challenges_vec = Some(vec![claim_point.to_vec()]);
        let g2_challenges = &claim_point[..self.num_dataparallel_vars];
        let g1_challenges = &claim_point[self.num_dataparallel_vars..];
        self.g1_challenges_vec = Some(vec![g1_challenges.to_vec()]);

        if self.num_dataparallel_vars > 0 {
            let beta_g2 = BetaValues::new(g2_challenges.iter().copied().enumerate().collect());
            self.beta_g2_vec = Some(vec![beta_g2]);
        }

        self.source_mle.index_mle_indices(0);
        Ok(())
    }

    fn initialize_rlc(&mut self, random_coefficients: &[F], claims: &[&RawClaim<F>]) {
        assert_eq!(random_coefficients.len(), claims.len());

        // Split all of the claimed challenges into whether they are claimed
        // challenges on the dataparallel variables or not.
        self.challenges_vec = Some(
            claims
                .iter()
                .map(|claim| claim.get_point().to_vec())
                .collect_vec(),
        );
        let (g2_challenges_vec, g1_challenges_vec): (Vec<&[F]>, Vec<&[F]>) = claims
            .iter()
            .map(|claim| claim.get_point().split_at(self.num_dataparallel_vars))
            .unzip();
        self.g1_challenges_vec = Some(
            g1_challenges_vec
                .into_iter()
                .map(|challenges| challenges.to_vec())
                .collect_vec(),
        );

        if self.num_dataparallel_vars > 0 {
            let beta_g2_vec = g2_challenges_vec
                .iter()
                .map(|g2_challenges| {
                    BetaValues::new(g2_challenges.iter().copied().enumerate().collect())
                })
                .collect();
            self.beta_g2_vec = Some(beta_g2_vec);
        }
        self.source_mle.index_mle_indices(0);
    }

    fn compute_round_sumcheck_message(
        &mut self,
        round_index: usize,
        random_coefficients: &[F],
    ) -> Result<Vec<F>, LayerError> {
        match round_index.cmp(&self.num_dataparallel_vars) {
            // Dataparallel phase.
            Ordering::Less => {
                let sumcheck_message = compute_sumcheck_message_data_parallel_identity_gate(
                    &self.source_mle,
                    &self.wiring,
                    self.num_dataparallel_vars - round_index,
                    &self
                        .challenges_vec
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|claim| &claim[round_index..])
                        .collect_vec(),
                    &self
                        .beta_g2_vec
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(random_coefficients)
                        .map(|(beta_values, random_coeff)| {
                            *random_coeff
                                * beta_values
                                    .updated_values
                                    .values()
                                    .fold(F::ONE, |acc, elem| acc * elem)
                        })
                        .collect_vec(),
                )
                .unwrap();
                Ok(sumcheck_message)
            }
            _ => {
                if round_index == self.num_dataparallel_vars {
                    match global_claim_agg_strategy() {
                        ClaimAggregationStrategy::Interpolative => {
                            // We compute the singular fully bound value for
                            let beta_g2_fully_bound = if self.num_dataparallel_vars > 0 {
                                self.beta_g2_vec.as_ref().unwrap()[0]
                                    .updated_values
                                    .values()
                                    .fold(F::ONE, |acc, val| acc * *val)
                            } else {
                                F::ONE
                            };

                            self.init_phase_1(
                                &self.g1_challenges_vec.as_ref().unwrap()[0].clone(),
                                beta_g2_fully_bound,
                            );
                        }
                        ClaimAggregationStrategy::RLC => {
                            let random_coefficients = if self.num_dataparallel_vars > 0 {
                                random_coefficients
                                    .iter()
                                    .zip(self.beta_g2_vec.as_ref().unwrap())
                                    .map(|(random_coeff, beta_values)| {
                                        if self.num_dataparallel_vars > 0 {
                                            beta_values
                                                .updated_values
                                                .values()
                                                .fold(F::ONE, |acc, val| acc * *val)
                                                * random_coeff
                                        } else {
                                            F::ONE * random_coeff
                                        }
                                    })
                                    .collect_vec()
                            } else {
                                random_coefficients.to_vec()
                            };

                            self.init_phase_1_rlc(
                                &self
                                    .g1_challenges_vec
                                    .as_ref()
                                    .unwrap()
                                    .clone()
                                    .iter()
                                    .map(|challenge| challenge.as_slice())
                                    .collect_vec(),
                                &random_coefficients,
                            );
                        }
                    }
                }

                let mles: Vec<&DenseMle<F>> =
                    vec![&self.a_hg_mle_phase_1.as_ref().unwrap(), &self.source_mle];
                let independent_variable = mles
                    .iter()
                    .map(|mle| mle.mle_indices().contains(&MleIndex::Indexed(round_index)))
                    .reduce(|acc, item| acc | item)
                    .unwrap();
                let sumcheck_evals =
                    evaluate_mle_product_no_beta_table(&mles, independent_variable, mles.len())
                        .unwrap();
                Ok(sumcheck_evals.0)
            }
        }
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        if round_index < self.num_dataparallel_vars {
            self.beta_g2_vec
                .as_mut()
                .unwrap()
                .iter_mut()
                .for_each(|beta| {
                    beta.beta_update(round_index, challenge);
                });
            self.source_mle.fix_variable(round_index, challenge);

            Ok(())
        } else {
            if self.num_dataparallel_vars > 0 {
                self.beta_g2_vec.as_ref().unwrap().iter().for_each(|beta| {
                    assert!(beta.unbound_values.is_empty());
                })
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
        let beta_bound = if self.num_dataparallel_vars != 0 {
            let g2_challenges = claim_challenges[..self.num_dataparallel_vars].to_vec();
            BetaValues::compute_beta_over_two_challenges(
                &g2_challenges,
                &round_challenges[..self.num_dataparallel_vars],
            )
        } else {
            F::ONE
        };
        let f_1_gu = compute_fully_bound_identity_gate_function(
            &round_challenges[self.num_dataparallel_vars..],
            &[&claim_challenges[self.num_dataparallel_vars..]],
            &self.wiring,
            &[beta_bound],
        );

        PostSumcheckLayer(vec![Product::<F, F>::new(
            &[self.source_mle.clone()],
            f_1_gu,
        )])
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {
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

/// identity gate struct wiring is used to be able to select specific elements
/// from an MLE (equivalent to an add gate with a zero mle ref)
#[derive(Error, Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct IdentityGate<F: Field> {
    /// layer id that this gate is found
    layer_id: LayerId,
    /// we only need a single incoming gate and a single outgoing gate so this
    /// is a tuple of 2 integers representing which label maps to which Tuples
    /// are of form `(dest_idx, src_idx)`.
    wiring: Vec<(u32, u32)>,
    /// the mle ref in question from which we are selecting specific indices
    source_mle: DenseMle<F>,
    /// The [BetaValues] struct which enumerates the incoming claim's challenge
    /// points on the dataparallel vars of the MLE
    beta_g2_vec: Option<Vec<BetaValues<F>>>,
    /// The nondataparallel claim points in the layer.
    g1_challenges_vec: Option<Vec<Vec<F>>>,
    /// The claim points in the layer.
    challenges_vec: Option<Vec<Vec<F>>>,
    /// The MLE initialized in phase 1, which contains the beta values over
    /// `g1_challenges` folded into the wiring function.
    a_hg_mle_phase_1: Option<DenseMle<F>>,
    /// The total number of variables in the layer.
    total_num_vars: usize,
    /// The number of vars representing the number of "dataparallel" copies of
    /// the circuit.
    num_dataparallel_vars: usize,
}

impl<F: Field> IdentityGate<F> {
    /// Create a new [IdentityGate] struct.
    pub fn new(
        layer_id: LayerId,
        wiring: Vec<(u32, u32)>,
        mle: DenseMle<F>,
        total_num_vars: usize,
        num_dataparallel_vars: usize,
    ) -> IdentityGate<F> {
        IdentityGate {
            layer_id,
            wiring,
            source_mle: mle,
            beta_g2_vec: None,
            a_hg_mle_phase_1: None,
            total_num_vars,
            num_dataparallel_vars,
            g1_challenges_vec: None,
            challenges_vec: None,
        }
    }

    fn append_leaf_mles_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<F>) {
        assert_eq!(self.source_mle.len(), 1);
        transcript_writer.append("Fully bound source MLE", self.source_mle.first());
    }

    /// Initialize the bookkeeping table necessary for phase 1, which is the
    /// binding of the non-dataparallel variables in the source MLE. This is the
    /// initialization function used when we are doing interpolative claim
    /// aggregation.
    ///
    /// For the random coefficients, we simply use the fully bound value of
    /// beta_g2 since this is the value that scales all of the sumcheck
    /// evaluations.
    fn init_phase_1(&mut self, challenge: &[F], fully_bound_beta_g2: F) {
        let a_hg_mle_vec = fold_wiring_into_beta_mle_identity_gate(
            &self.wiring,
            &[challenge],
            self.source_mle.num_free_vars(),
            &[fully_bound_beta_g2],
        );
        let mut a_hg_mle = DenseMle::new_from_raw(a_hg_mle_vec, self.layer_id());
        a_hg_mle.index_mle_indices(self.num_dataparallel_vars);

        self.a_hg_mle_phase_1 = Some(a_hg_mle);
    }

    /// Initialize the bookkeeping table necessary for phase 1, which is the
    /// binding of the non-dataparallel variables in the source MLE. This is the
    /// initialization function used when we are doing RLC claim
    /// aggregation.
    fn init_phase_1_rlc(&mut self, challenges: &[&[F]], random_coefficients: &[F]) {
        let a_hg_mle_vec = fold_wiring_into_beta_mle_identity_gate(
            &self.wiring,
            challenges,
            self.source_mle.num_free_vars(),
            random_coefficients,
        );
        let mut a_hg_mle = DenseMle::new_from_raw(a_hg_mle_vec, self.layer_id());
        a_hg_mle.index_mle_indices(self.num_dataparallel_vars);
        self.a_hg_mle_phase_1 = Some(a_hg_mle);
    }
}

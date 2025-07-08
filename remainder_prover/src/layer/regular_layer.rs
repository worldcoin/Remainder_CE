//! The implementation of `RegularLayer`

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use remainder_shared_types::{
    config::{global_config::global_claim_agg_strategy, ClaimAggregationStrategy},
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    claims::{Claim, ClaimError, RawClaim},
    expression::{
        circuit_expr::{filter_bookkeeping_table, ExprDescription},
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
        verifier_expr::VerifierExpr,
    },
    layer::{Layer, LayerId, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{betavalues::BetaValues, dense::DenseMle, mle_description::MleDescription, Mle},
    sumcheck::{evaluate_at_a_point, get_round_degree},
};

use super::{
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::PostSumcheckLayer,
};

use super::{LayerDescription, VerifierLayer};

use anyhow::{anyhow, Ok, Result};

/// The most common implementation of [crate::layer::Layer].
///
/// A regular layer is made up of a structured polynomial relationship between
/// MLEs of previous layers.
///
/// Proofs are generated with the Sumcheck protocol.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct RegularLayer<F: Field> {
    /// This layer's ID.
    id: LayerId,

    /// The polynomial expression defining this layer.
    /// It includes information on how this layer relates to the others.
    pub(crate) expression: Expression<F, ProverExpr>,

    /// Stores the indices of the sumcheck rounds in this GKR layer so we
    /// only produce sumcheck proofs over those. When we use interpolative
    /// claim aggregation, this is all of the nonlinear variables in the
    /// expression. When we use RLC claim aggregation, this is all of the
    /// variables in the expression.
    sumcheck_rounds: Vec<usize>,

    /// Stores the beta values associated with the `expression`.
    /// Initially set to `None`. Computed during initialization.
    beta_vals_vec: Option<Vec<BetaValues<F>>>,
}

impl<F: Field> RegularLayer<F> {
    /// Creates a new `RegularLayer` from an `Expression` and a `LayerId`
    ///
    /// The `Expression` is the relationship this `Layer` proves
    /// and the `LayerId` is the location of this `Layer` in the overall circuit
    pub fn new_raw(id: LayerId, mut expression: Expression<F, ProverExpr>) -> Self {
        // Compute nonlinear rounds from `expression`
        expression.index_mle_indices(0);
        let sumcheck_rounds = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => expression.get_all_nonlinear_rounds(),
            ClaimAggregationStrategy::RLC => expression.get_all_rounds(),
        };
        RegularLayer {
            id,
            expression,
            sumcheck_rounds,
            beta_vals_vec: None,
        }
    }

    /// Returns a reference to the expression that this layer is proving.
    pub fn get_expression(&self) -> &Expression<F, ProverExpr> {
        &self.expression
    }

    /// Traverse the fully-bound `self.expression` and append all MLE values
    /// to the trascript.
    pub fn append_leaf_mles_to_transcript(
        &self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<()> {
        let mut observer_fn = |expr_node: &ExpressionNode<F, ProverExpr>,
                               mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
         -> Result<()> {
            match expr_node {
                ExpressionNode::Mle(mle_vec_index) => {
                    let mle: &DenseMle<F> = &mle_vec[mle_vec_index.index()];
                    let val = mle.mle.value();
                    transcript_writer.append("Leaf MLE value", val);
                    Ok(())
                }
                ExpressionNode::Product(mle_vec_indices) => {
                    for mle_vec_index in mle_vec_indices {
                        let mle = &mle_vec[mle_vec_index.index()];
                        let eval = mle.mle.value();
                        transcript_writer.append("Product MLE value", eval);
                    }
                    Ok(())
                }
                ExpressionNode::Constant(_)
                | ExpressionNode::Scaled(_, _)
                | ExpressionNode::Sum(_, _)
                | ExpressionNode::Selector(_, _, _) => Ok(()),
            }
        };

        let _ = self.expression.traverse(&mut observer_fn);

        Ok(())
    }
}

impl<F: Field> Layer<F> for RegularLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn prove(
        &mut self,
        claims: &[&RawClaim<F>],
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<()> {
        info!("Proving a GKR Layer.");
        // Initialize tables and pre-fix variables.
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

        let mut previous_round_message = vec![claims
            .iter()
            .zip(&random_coefficients)
            .fold(F::ZERO, |acc, (claim, random_coeff)| {
                acc + claim.get_eval() * random_coeff
            })];
        let mut previous_challenge = F::ZERO;

        let layer_id = self.layer_id();
        for round_index in self.sumcheck_rounds.clone() {
            // First compute the appropriate number of univariate evaluations for this round.
            let prover_sumcheck_message =
                self.compute_round_sumcheck_message(round_index, &random_coefficients)?;
            // In debug mode, catch sumcheck round errors from the prover side.
            debug_assert_eq!(
                evaluate_at_a_point(&previous_round_message, previous_challenge).unwrap(),
                prover_sumcheck_message[0] + prover_sumcheck_message[1],
                "failed at round {round_index}, layer {layer_id}",
            );
            // Append the evaluations to the transcript.
            // Since the verifier can deduce g_i(0) by computing claim - g_i(1), the prover does not send g_i(0)
            transcript_writer.append_elements("Sumcheck message", &prover_sumcheck_message[1..]);
            // Sample the challenge
            let challenge = transcript_writer.get_challenge("Sumcheck challenge");
            // "Bind" the challenge to the expression at this point.
            self.bind_round_variable(round_index, challenge)?;
            // For debug mode, update the previous message and challenge for the purpose
            // of checking whether these still pass the sumcheck round checks.
            previous_round_message = prover_sumcheck_message;
            previous_challenge = challenge;
        }

        // By now, `self.expression` should be fully bound.
        assert_eq!(self.expression.get_expression_num_free_variables(), 0);

        // Append the values of the leaf MLEs to the transcript.
        self.append_leaf_mles_to_transcript(transcript_writer)?;

        Ok(())
    }

    /// Initialize all necessary information in order to start sumcheck within a
    /// layer of GKR. This includes pre-fixing all of the rounds within the
    /// layer which are linear, and then appropriately initializing the
    /// necessary beta values over the nonlinear rounds.
    fn initialize(&mut self, claim_point: &[F]) -> Result<()> {
        let expression = &mut self.expression;
        let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
        let expression_linear_indices = expression.get_all_linear_rounds();

        // For each of the linear indices in the expression, we can fix the
        // variable at that index of the expression, so that now the only
        // unbound indices are the nonlinear indices.
        expression_linear_indices
            .iter()
            .sorted()
            .for_each(|round_idx| {
                expression.fix_variable_at_index(*round_idx, claim_point[*round_idx]);
            });

        // We need the beta values over the nonlinear indices of the expression,
        // so we grab the claim points that are over these nonlinear indices and
        // then initialize the betavalues struct over them.
        let betavec = expression_nonlinear_indices
            .iter()
            .map(|idx| (*idx, claim_point[*idx]))
            .collect_vec();
        let newbeta = BetaValues::new(betavec);
        self.beta_vals_vec = Some(vec![newbeta]);

        Ok(())
    }

    fn initialize_rlc(&mut self, _random_coefficients: &[F], claims: &[&RawClaim<F>]) {
        // We need the beta values over all the indices of the expression, as we
        // cannot perform the linear round optimization with RLC claim agg since
        // we have multiple points to bind each MLE to.
        let expression = &mut self.expression;
        let expression_all_indices = expression.get_all_rounds();

        let beta_vals_vec = claims
            .iter()
            .map(|claim| {
                let claim_point = claim.get_point();
                let betavec = expression_all_indices
                    .iter()
                    .map(|idx| (*idx, claim_point[*idx]))
                    .collect_vec();
                BetaValues::new(betavec)
            })
            .collect();
        self.beta_vals_vec = Some(beta_vals_vec);
    }

    fn compute_round_sumcheck_message(
        &mut self,
        round_index: usize,
        random_coefficients: &[F],
    ) -> Result<Vec<F>> {
        // Grabs the expression/beta table.
        let expression = &self.expression;
        let newbeta = &self.beta_vals_vec;

        // Grabs the degree of univariate polynomial we are sending over.
        let degree = get_round_degree(expression, round_index);

        // Computes the sumcheck message using the beta cascade algorithm.
        let prover_sumcheck_message = expression.evaluate_sumcheck_beta_cascade(
            &newbeta.as_ref().unwrap().iter().collect_vec(),
            random_coefficients,
            round_index,
            degree,
        );

        Ok(prover_sumcheck_message.0)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<()> {
        // Grabs the expression/beta table.
        let expression = &mut self.expression;
        let beta_vals_vec = &mut self.beta_vals_vec;

        // Update the bookkeeping tables as necessary.
        expression.fix_variable(round_index, challenge);
        beta_vals_vec
            .as_mut()
            .unwrap()
            .iter_mut()
            .for_each(|beta_vals| {
                beta_vals.beta_update(round_index, challenge);
            });

        Ok(())
    }

    /// Returns the round indices (with respect to the indices of all relevant
    /// variables within the layer) which are nonlinear. For example, if the
    /// current layer's [Expression] looks something like
    /// V_{i + 1}(x_1, x_2, x_3) * V_{i + 2}(x_1, x_2)
    /// then the `sumcheck_round_indices` of this layer would be [1, 2].
    fn sumcheck_round_indices(&self) -> Vec<usize> {
        self.sumcheck_rounds.clone()
    }

    fn max_degree(&self) -> usize {
        &self.expression.get_max_degree() + 1
    }

    /// Get the [PostSumcheckLayer] for a regular layer, which represents the fully bound expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[&[F]],
        random_coefficients: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let sumcheck_round_indices = self.sumcheck_round_indices();
        // Filter the claim to get the values of the claim pertaining to the nonlinear rounds.
        let sumcheck_claim_points_vec = claim_challenges
            .iter()
            .map(|claim_challenge| {
                claim_challenge
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, point)| {
                        if sumcheck_round_indices.contains(&idx) {
                            Some(*point)
                        } else {
                            None
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Compute beta over these and the sumcheck challenges.
        let rlc_beta = sumcheck_claim_points_vec
            .iter()
            .zip(random_coefficients)
            .fold(F::ZERO, |acc, (elem, random_coeff)| {
                assert_eq!(round_challenges.len(), elem.len());
                let fully_bound_beta =
                    BetaValues::compute_beta_over_two_challenges(round_challenges, elem);
                acc + fully_bound_beta * random_coeff
            });

        self.expression.get_post_sumcheck_layer(rlc_beta)
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
        // First off, parse the expression that is associated with the layer.
        // Next, get to the actual claims that are generated by each expression and grab them
        // Return basically a list of (usize, Claim)
        let layerwise_expr = &self.expression;

        let mut claims: Vec<Claim<F>> = Vec::new();

        // Define how to parse the expression tree.
        // Basically we just want to go down it and pass up claims.
        // We can only add a new claim if we see an MLE with all its indices
        // bound.
        let mut observer_fn = |expr: &ExpressionNode<F, ProverExpr>,
                               mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
         -> Result<()> {
            match expr {
                ExpressionNode::Mle(mle_vec_idx) => {
                    let mle = mle_vec_idx.get_mle(mle_vec);

                    let fixed_mle_indices = mle
                        .mle_indices
                        .iter()
                        .map(|index| index.val().ok_or(anyhow!(ClaimError::MleRefMleError)))
                        .collect::<Result<Vec<_>>>()?;

                    // Grab the layer ID (i.e. MLE index) which this mle refers to
                    let mle_layer_id = mle.layer_id();

                    let claimed_value = mle.value();

                    // Note: No need to append claim values here.
                    // We already appended them when evaluating the
                    // expression for sumcheck.

                    // Construct the claim
                    let claim = Claim::new(
                        fixed_mle_indices,
                        claimed_value,
                        self.layer_id(),
                        mle_layer_id,
                    );

                    // Push it into the list of claims
                    claims.push(claim);
                }
                ExpressionNode::Product(mle_vec_indices) => {
                    for mle_vec_index in mle_vec_indices {
                        let mle = mle_vec_index.get_mle(mle_vec);
                        let fixed_mle_indices = mle
                            .mle_indices
                            .iter()
                            .map(|index| index.val().ok_or(anyhow!(ClaimError::MleRefMleError)))
                            .collect::<Result<Vec<_>>>()?;

                        // Grab the layer ID (i.e. MLE index) which this mle refers to
                        let mle_layer_id = mle.layer_id();

                        let claimed_value = mle.value();

                        // Note: No need to append the claim value to the transcript here. We
                        // already appended when evaluating the expression for sumcheck.

                        // Construct the claim
                        // need to populate the claim with the mle ref we are grabbing the claim from
                        let claim = Claim::new(
                            fixed_mle_indices,
                            claimed_value,
                            self.layer_id(),
                            mle_layer_id,
                        );

                        // Push it into the list of claims
                        claims.push(claim);
                    }
                }
                _ => {}
            }
            Ok(())
        };

        // Apply the observer function from above onto the expression
        layerwise_expr.traverse(&mut observer_fn)?;

        Ok(claims)
    }
}

/// The circuit description counterpart of a [RegularLayer].
#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
#[serde(bound = "F: Field")]
pub struct RegularLayerDescription<F: Field> {
    /// This layer's ID.
    id: LayerId,

    /// A structural description of the polynomial expression defining this
    /// layer. The leaves of the expression describe the MLE characteristics
    /// without storing any values.
    expression: Expression<F, ExprDescription>,
}

impl<F: Field> RegularLayerDescription<F> {
    /// Generates a new [RegularLayerDescription] given raw data.
    pub fn new_raw(id: LayerId, expression: Expression<F, ExprDescription>) -> Self {
        Self { id, expression }
    }
}

/// The verifier's counterpart of a [RegularLayer].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct VerifierRegularLayer<F: Field> {
    /// This layer's ID.
    id: LayerId,

    /// A fully-bound expression defining the layer.
    expression: Expression<F, VerifierExpr>,
}

impl<F: Field> VerifierRegularLayer<F> {
    /// Generates a new [VerifierRegularLayer] given raw data.
    pub(crate) fn new_raw(id: LayerId, expression: Expression<F, VerifierExpr>) -> Self {
        Self { id, expression }
    }
}

impl<F: Field> LayerDescription<F> for RegularLayerDescription<F> {
    type VerifierLayer = VerifierRegularLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&MleDescription<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) {
        let mut expression_nodes_to_compile =
            HashMap::<&ExpressionNode<F, ExprDescription>, Vec<(Vec<bool>, Vec<bool>)>>::new();

        mle_outputs_necessary
            .iter()
            .for_each(|mle_output_necessary| {
                let prefix_bits = mle_output_necessary.prefix_bits();
                let mut unfiltered_prefix_bits: Vec<bool> = vec![];
                let expression_node_to_compile = prefix_bits.iter().fold(
                    &self.expression.expression_node,
                    |acc, bit| match acc {
                        ExpressionNode::Selector(_mle_index, lhs, rhs) => {
                            if *bit {
                                rhs
                            } else {
                                lhs
                            }
                        }
                        _ => {
                            unfiltered_prefix_bits.push(*bit);
                            acc
                        }
                    },
                );
                if expression_nodes_to_compile.contains_key(expression_node_to_compile) {
                    expression_nodes_to_compile
                        .get_mut(expression_node_to_compile)
                        .unwrap()
                        .push((unfiltered_prefix_bits.clone(), prefix_bits));
                } else {
                    expression_nodes_to_compile.insert(
                        expression_node_to_compile,
                        vec![(unfiltered_prefix_bits.clone(), prefix_bits)],
                    );
                }
            });

        expression_nodes_to_compile
            .iter()
            .for_each(|(expression_node, prefix_bit_vec)| {
                let full_bookkeeping_table = expression_node
                    .compute_bookkeeping_table(circuit_map)
                    .unwrap();
                prefix_bit_vec
                    .iter()
                    .for_each(|(unfiltered_prefix_bits, prefix_bits)| {
                        let filtered_table = filter_bookkeeping_table(
                            &full_bookkeeping_table,
                            unfiltered_prefix_bits,
                        );
                        circuit_map.add_node(
                            CircuitLocation::new(self.layer_id(), prefix_bits.clone()),
                            filtered_table,
                        );
                    });
            });
    }

    fn verify_rounds(
        &self,
        claims: &[&RawClaim<F>],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>> {
        let rounds_sumchecked_over = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => self.expression.get_all_nonlinear_rounds(),
            ClaimAggregationStrategy::RLC => self.expression.get_all_rounds(),
        };

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

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
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

        // For round 1 <= i <= n, perform the check:
        for round_index in &rounds_sumchecked_over {
            let degree = self.expression.get_round_degree(*round_index);

            // Receive `g_i(x)` from the Prover.
            // Since we are using an evaluation representation for polynomials,
            // the degree check is implicit: the verifier is requesting
            // `degree + 1` evaluations, ensuring that `g_i` is of degree
            // at most `degree`. If the prover appended more evaluations,
            // there will be a transcript read error later on in the proving
            // process which will result in the proof not verifying.
            // Furthermore, since the verifier can deduce g_i(0) by computing `claim - g_i(1)`,
            // the prover does not include g_i(0) in the message. Instead, the verifier
            // reserves the spot of g_i(0) when reading from the transcript, and compute g_i(0)
            // afterwards.
            // TODO(Makis):
            //   1. Modify the Transcript interface to catch any errors sooner.
            //   2. This line is assuming a representation for the polynomial!
            //   We should hide that under another function whose job is to take
            //   the trascript reader and read the polynomial in whatever
            //   representation is being used.
            let mut g_cur_round: Vec<_> = [Ok(F::from(0))]
                .into_iter()
                .chain((0..degree).map(|_| transcript_reader.consume_element("Sumcheck message")))
                .collect::<Result<_, _>>()?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_challenge("Sumcheck challenge")?;

            // TODO(Makis): After refactoring `SumcheckEvals` to be a
            // representation of a univariate polynomial, `evaluate_at_a_point`
            // should just be a method.
            // Compute:
            //       `g_i(0) = g_{i - 1}(r_{i-1}) - g_i(1)`
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, F::ONE).unwrap();
            g_cur_round[0] = g_prev_r_prev - g_i_one;

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // TODO(Makis): Add check that `expr` is on the same number of total vars.
        let num_vars = claims[0].get_num_vars();

        // Build an indicator vector for linear indices.
        let mut var_is_linear: Vec<bool> = vec![true; num_vars];
        if global_claim_agg_strategy() == ClaimAggregationStrategy::Interpolative {
            for idx in &rounds_sumchecked_over {
                var_is_linear[*idx] = false;
            }
        }
        // Build point interlacing linear-round challenges with nonlinear-round
        // challenges.
        let mut nonlinear_idx = 0;
        let point: &Vec<F> = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => &(0..num_vars)
                .map(|idx| {
                    if var_is_linear[idx] {
                        claims[0].get_point()[idx]
                    } else {
                        let r = challenges[nonlinear_idx];
                        nonlinear_idx += 1;
                        r
                    }
                })
                .collect(),
            ClaimAggregationStrategy::RLC => &challenges,
        };

        let verifier_layer = self
            .convert_into_verifier_layer(
                point,
                &claims.iter().map(|claim| claim.get_point()).collect_vec(),
                transcript_reader,
            )
            .unwrap();

        // Compute `P(r_1, ..., r_n)` over all challenge points (linear and
        // non-linear).
        // The MLE values are retrieved from the transcript.
        let expr_value_at_challenge_point = verifier_layer.expression.evaluate()?;

        let beta_fn_evaluated_at_challenge_point = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                // Compute `\beta((r_1, ..., r_n), (u_1, ..., u_n))`.
                let claim_nonlinear_vals: Vec<F> = rounds_sumchecked_over
                    .iter()
                    .map(|idx| (claims[0].get_point()[*idx]))
                    .collect();
                debug_assert_eq!(claim_nonlinear_vals.len(), challenges.len());
                BetaValues::compute_beta_over_two_challenges(&claim_nonlinear_vals, &challenges)
            }
            ClaimAggregationStrategy::RLC => random_coefficients.iter().zip(claims).fold(
                F::ZERO,
                |acc, (random_coeff, claim)| {
                    acc + *random_coeff
                        * BetaValues::compute_beta_over_two_challenges(
                            claim.get_point(),
                            &challenges,
                        )
                },
            ),
        };

        // Evalute `g_n(r_n)`.
        // Note: If there were no nonlinear rounds, this value reduces to
        // `claim.get_result()` due to how we initialized `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        // Final check:
        // `\sum_{b_2} \sum_{b_4} P(g_1, b_2, g_3, b_4) * \beta( (b_2, b_4), (g_2, g_4) )`.
        // P(g_1, challenge[0], g_3, challenge[0]) * \beta( challenge, (g_2, g_4) )
        // `g_n(r_n) == P(r_1, ..., r_n) * \beta(r_1, ..., r_n, g_1, ..., g_n)`.
        if g_final_r_final != expr_value_at_challenge_point * beta_fn_evaluated_at_challenge_point {
            return Err(anyhow!(VerificationError::SumcheckFailed));
        }

        Ok(VerifierLayerEnum::Regular(verifier_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => self.expression.get_all_nonlinear_rounds(),
            ClaimAggregationStrategy::RLC => self.expression.get_all_rounds(),
        }
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_challenges: &[F],
        _claim_point: &[&[F]],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer> {
        let verifier_expr = self
            .expression
            .bind(sumcheck_challenges, transcript_reader)?;

        let verifier_layer = VerifierRegularLayer::new_raw(self.layer_id(), verifier_expr);
        Ok(verifier_layer)
    }

    /// Get the [PostSumcheckLayer] for a [RegularLayerDescription], which represents the description of a fully bound expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[&[F]],
        random_coefficients: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        let sumcheck_round_indices = self.sumcheck_round_indices();
        // Filter the claim to get the values of the claim pertaining to the nonlinear rounds.
        let sumcheck_claim_points_vec = claim_challenges
            .iter()
            .map(|claim_challenge| {
                claim_challenge
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, point)| {
                        if sumcheck_round_indices.contains(&idx) {
                            Some(*point)
                        } else {
                            None
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Compute beta over these and the sumcheck challenges.
        let rlc_beta = sumcheck_claim_points_vec
            .iter()
            .zip(random_coefficients)
            .fold(F::ZERO, |acc, (elem, random_coeff)| {
                assert_eq!(round_challenges.len(), elem.len());
                let fully_bound_beta =
                    BetaValues::compute_beta_over_two_challenges(round_challenges, elem);
                acc + fully_bound_beta * random_coeff
            });

        // Compute the fully bound challenges, which include those pre-fixed for linear rounds
        // and the sumcheck rounds.

        let all_bound_challenges = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                assert_eq!(claim_challenges.len(), 1);
                let mut sumcheck_round_index_counter = 0;
                let all_chals = (0..claim_challenges[0].len())
                    .map(|idx| {
                        if sumcheck_round_indices.contains(&idx) {
                            let chal = round_challenges[sumcheck_round_index_counter];
                            sumcheck_round_index_counter += 1;
                            chal
                        } else {
                            claim_challenges[0][idx]
                        }
                    })
                    .collect_vec();
                assert_eq!(sumcheck_round_index_counter, sumcheck_round_indices.len());
                all_chals
            }
            ClaimAggregationStrategy::RLC => round_challenges.to_vec(),
        };

        self.expression
            .get_post_sumcheck_layer(rlc_beta, &all_bound_challenges)
    }

    fn max_degree(&self) -> usize {
        self.expression.get_max_degree() + 1
    }

    fn get_circuit_mles(&self) -> Vec<&MleDescription<F>> {
        self.expression.get_circuit_mles()
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let prover_expr = self.expression.into_prover_expression(circuit_map);
        let regular_layer = RegularLayer::new_raw(self.layer_id(), prover_expr);
        regular_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.expression.index_mle_vars(start_index);
    }
}

impl<F: Field> VerifierLayer<F> for VerifierRegularLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>> {
        let expr = &self.expression;

        // Define how to parse the expression tree
        // - Basically we just want to go down it and pass up claims
        // - We can only add a new claim if we see an MLE with all its indices bound

        let mut claims: Vec<Claim<F>> = Vec::new();

        let mut observer_fn = |exp: &ExpressionNode<F, VerifierExpr>,
                               _mle_vec: &<VerifierExpr as ExpressionType<F>>::MleVec|
         -> Result<()> {
            match exp {
                ExpressionNode::Mle(verifier_mle) => {
                    let fixed_mle_indices = verifier_mle
                        .var_indices()
                        .iter()
                        .map(|index| index.val().ok_or(anyhow!(ClaimError::MleRefMleError)))
                        .collect::<Result<Vec<_>>>()?;

                    // Grab the layer ID (i.e. MLE index) which this mle refers to
                    let mle_layer_id = verifier_mle.layer_id();

                    // Grab the actual value that the claim is supposed to evaluate to
                    let claimed_value = verifier_mle.value();

                    // Construct the claim
                    let claim: Claim<F> = Claim::new(
                        fixed_mle_indices,
                        claimed_value,
                        self.layer_id(),
                        mle_layer_id,
                    );

                    // Push it into the list of claims
                    claims.push(claim);
                }
                ExpressionNode::Product(verifier_mle_vec) => {
                    for verifier_mle in verifier_mle_vec {
                        let fixed_mle_indices = verifier_mle
                            .var_indices()
                            .iter()
                            .map(|index| index.val().ok_or(anyhow!(ClaimError::MleRefMleError)))
                            .collect::<Result<Vec<_>>>()?;

                        // Grab the layer ID (i.e. MLE index) which this mle refers to
                        let mle_layer_id = verifier_mle.layer_id();

                        let claimed_value = verifier_mle.value();

                        // Construct the claim
                        let claim: Claim<F> = Claim::new(
                            fixed_mle_indices,
                            claimed_value,
                            self.layer_id(),
                            mle_layer_id,
                        );

                        // Push it into the list of claims
                        claims.push(claim);
                    }
                }
                _ => {}
            }
            Ok(())
        };

        // Apply the observer function from above onto the expression
        expr.traverse(&mut observer_fn)?;

        Ok(claims)
    }
}

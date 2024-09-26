//! The implementation of `RegularLayer`

/// The implementation of claim handling helpers for `RegularLayer`
pub mod claims;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    builders::layer_builder::LayerBuilder,
    claims::Claim,
    expression::{
        circuit_expr::{filter_bookkeeping_table, CircuitExpr, CircuitMle},
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
        verifier_expr::VerifierExpr,
    },
    layer::{Layer, LayerError, LayerId, VerificationError},
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{betavalues::BetaValues, dense::DenseMle},
    sumcheck::{compute_sumcheck_message_beta_cascade, evaluate_at_a_point, get_round_degree},
};

use super::{
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::PostSumcheckLayer,
};

use super::{CircuitLayer, VerifierLayer};

/// The most common implementation of [crate::layer::Layer].
///
/// A layer is made up of a structured polynomial relationship between MLEs of
/// previous layers.
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

    /// Stores the indices of the non-linear rounds in this GKR layer so we
    /// only produce sumcheck proofs over those.
    nonlinear_rounds: Option<Vec<usize>>,

    /// Store the beta values associated with an expression.
    beta_vals: Option<BetaValues<F>>,
}

/// The circuit description counterpart of a [RegularLayer].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct CircuitRegularLayer<F: Field> {
    /// This layer's ID.
    id: LayerId,

    /// A structural description of the polynomial expression defining this
    /// layer. Leaves of the expression describe the MLE characteristics without
    /// storing any values.
    expression: Expression<F, CircuitExpr>,
}

impl<F: Field> CircuitRegularLayer<F> {
    /// To be used internally only!
    /// Generates a new [CircuitRegularLayer] given raw data.
    pub fn new_raw(id: LayerId, expression: Expression<F, CircuitExpr>) -> Self {
        Self { id, expression }
    }
}

/// The verifier counterpart of a [RegularLayer].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct VerifierRegularLayer<F: Field> {
    /// This layer's ID.
    id: LayerId,

    /// A fully-bound expression defining the layer.
    expression: Expression<F, VerifierExpr>,
}

impl<F: Field> VerifierRegularLayer<F> {
    /// To be used internally only!
    /// Generates a new [VerifierRegularLayer] given raw data.
    pub(crate) fn new_raw(id: LayerId, expression: Expression<F, VerifierExpr>) -> Self {
        Self { id, expression }
    }
}

impl<F: Field> Layer<F> for RegularLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        info!("Proving a GKR Layer.");

        // Initialize tables and pre-fix variables.
        self.start_sumcheck(&claim)?;

        let nonlinear_rounds = self.nonlinear_rounds.take().unwrap();

        for round_index in &nonlinear_rounds {
            self.prove_nonlinear_round(transcript_writer, *round_index)?;
            // TODO(Makis): Add debug assertion that g_i(0) + g_i(1) == g_{i-1}(r_i).
        }

        // By now, `self.expression` should be fully bound.
        // TODO(Makis): Add assertion for that.

        // Append the values of the leaf MLEs to the transcript.
        self.append_leaf_mles_to_transcript(transcript_writer)?;

        Ok(())
    }

    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        let expression = &mut self.expression;
        let _expression_num_indices = expression.index_mle_indices(0);
        let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
        let expression_linear_indices = expression.get_all_linear_rounds();

        // for each of the linear indices in the expression, we can fix the variable at that index for
        // the expression, so that now the only unbound indices are the nonlinear indices.
        expression_linear_indices
            .iter()
            .sorted()
            .for_each(|round_idx| {
                expression.fix_variable_at_index(*round_idx, claim_point[*round_idx]);
            });

        // we need the beta values over the nonlinear indices of the expression, so we grab
        // the claim points that are over these nonlinear indices and then initialize the betavalues
        // struct over them.
        let betavec = expression_nonlinear_indices
            .iter()
            .map(|idx| (*idx, claim_point[*idx]))
            .collect_vec();
        let newbeta = BetaValues::new(betavec);
        self.beta_vals = Some(newbeta);

        // store the nonlinear rounds of the expression within the layer so that we know these are the
        // rounds we perform sumcheck over.
        self.nonlinear_rounds = Some(expression_nonlinear_indices);
        Ok(())
    }

    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError> {
        // Grabs the expression/beta table.
        let expression = &self.expression;
        let newbeta = &self.beta_vals;

        // Grabs the degree of univariate polynomial we are sending over.
        let degree = get_round_degree(expression, round_index);

        // Computes the sumcheck message using the beta cascade algorithm.
        let prover_sumcheck_message = compute_sumcheck_message_beta_cascade(
            expression,
            round_index,
            degree,
            newbeta.as_ref().unwrap(),
        )
        .unwrap();

        Ok(prover_sumcheck_message.0)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        // Grabs the expression/beta table.
        let expression = &mut self.expression;
        let newbeta = &mut self.beta_vals;

        // Update the bookkeeping tables as necessary.
        expression.fix_variable(round_index, challenge);
        newbeta
            .as_mut()
            .unwrap()
            .beta_update(round_index, challenge);

        Ok(())
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        self.nonlinear_rounds.clone().unwrap()
    }

    fn max_degree(&self) -> usize {
        &self.expression.get_max_degree() + 1
    }

    /// Get the [PostSumcheckLayer] for a regular layer, which represents the fully bound expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let nonlinear_round_indices = self.sumcheck_round_indices();
        // Filter the claim to get the values of the claim pertaining to the nonlinear rounds.
        let nonlinear_claim_points = claim_challenges
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                if nonlinear_round_indices.contains(&idx) {
                    Some(*point)
                } else {
                    None
                }
            })
            .collect_vec();
        assert_eq!(round_challenges.len(), nonlinear_claim_points.len());

        // Compute beta over these and the sumcheck challenges.
        let fully_bound_beta =
            BetaValues::compute_beta_over_two_challenges(round_challenges, &nonlinear_claim_points);

        self.expression.get_post_sumcheck_layer(fully_bound_beta)
    }
}

impl<F: Field> CircuitLayer<F> for CircuitRegularLayer<F> {
    type VerifierLayer = VerifierRegularLayer<F>;

    fn layer_id(&self) -> LayerId {
        self.id
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&CircuitMle<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) {
        let mut expression_nodes_to_compile =
            HashMap::<&ExpressionNode<F, CircuitExpr>, Vec<(Vec<bool>, Vec<bool>)>>::new();

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
                let full_bookkeeping_table = expression_node.compute_bookkeeping_table(circuit_map).unwrap();
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
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        let nonlinear_rounds = self.expression.get_all_nonlinear_rounds();

        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
        let mut g_prev_round = vec![claim.get_result()];

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = F::ZERO;

        // For round 1 <= i <= n, perform the check:
        for round_index in &nonlinear_rounds {
            let degree = self.expression.get_round_degree(*round_index);

            // Receive `g_i(x)` from the Prover.
            // Since we are using an evaluation representation for polynomials,
            // the degree check is implicit: the verifier is requesting
            // `degree + 1` evaluations, ensuring that `g_i` is of degree
            // at most `degree`. If the prover appended more evaluations,
            // there will be a transcript read error later on in the proving
            // process which will result in the proof not verifying.
            // TODO(Makis):
            //   1. Modify the Transcript interface to catch any errors sooner.
            //   2. This line is assuming a representation for the polynomial!
            //   We should hide that under another function whose job is to take
            //   the trascript reader and read the polynomial in whatever
            //   representation is being used.
            let g_cur_round = transcript_reader
                .consume_elements("Sumcheck message", degree + 1)
                .map_err(VerificationError::TranscriptError)?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_challenge("Sumcheck challenge")?;

            // TODO(Makis): After refactoring `SumcheckEvals` to be a
            // representation of a univariate polynomial, `evaluate_at_a_point`
            // should just be a method.
            // Verify that:
            //       `g_i(0) + g_i(1) == g_{i - 1}(r_{i-1})`
            let g_i_zero = evaluate_at_a_point(&g_cur_round, F::ZERO).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, F::ONE).unwrap();
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();

            if g_i_zero + g_i_one != g_prev_r_prev {
                return Err(VerificationError::SumcheckFailed);
            }

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // TODO(Makis): Add check that `expr` is on the same number of total vars.
        let num_vars = claim.get_num_vars();

        // Build an indicator vector for linear indices.
        let mut var_is_linear: Vec<bool> = vec![true; num_vars];
        for idx in &nonlinear_rounds {
            var_is_linear[*idx] = false;
        }

        // Build point interlacing linear-round challenges with nonlinear-round
        // challenges.
        let mut nonlinear_idx = 0;
        let point: Vec<F> = (0..num_vars)
            .map(|idx| {
                if var_is_linear[idx] {
                    claim.get_point()[idx]
                } else {
                    let r = challenges[nonlinear_idx];
                    nonlinear_idx += 1;
                    r
                }
            })
            .collect();

        let verifier_layer = self
            .convert_into_verifier_layer(&point, claim.get_point(), transcript_reader)
            .unwrap();

        // Compute `P(r_1, ..., r_n)` over all challenge points (linear and
        // non-linear).
        // The MLE values are retrieved from the transcript.
        let expr_value_at_challenge_point = verifier_layer.expression.evaluate()?;

        // Compute `\beta((r_1, ..., r_n), (u_1, ..., u_n))`.
        let claim_nonlinear_vals: Vec<F> = nonlinear_rounds
            .iter()
            .map(|idx| (claim.get_point()[*idx]))
            .collect();
        debug_assert_eq!(claim_nonlinear_vals.len(), challenges.len());

        let beta_fn_evaluated_at_challenge_point =
            BetaValues::compute_beta_over_two_challenges(&claim_nonlinear_vals, &challenges);

        // Evalute `g_n(r_n)`.
        // Note: If there were no nonlinear rounds, this value reduces to
        // `claim.get_result()` due to how we initialized `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        // dbg!(&challenges, &claim_nonlinear_vals);
        // dbg!(&g_final_r_final);
        // dbg!(&expr_value_at_challenge_point);
        // dbg!(&beta_fn_evaluated_at_challenge_point);
        // Final check:
        // `\sum_{b_2} \sum_{b_4} P(g_1, b_2, g_3, b_4) * \beta( (b_2, b_4), (g_2, g_4) )`.
        // P(g_1, challenge[0], g_3, challenge[0]) * \beta( challenge, (g_2, g_4) )
        // `g_n(r_n) == P(r_1, ..., r_n) * \beta(r_1, ..., r_n, g_1, ..., g_n)`.
        if g_final_r_final != expr_value_at_challenge_point * beta_fn_evaluated_at_challenge_point {
            return Err(VerificationError::SumcheckFailed);
        }

        Ok(VerifierLayerEnum::Regular(verifier_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        self.expression.get_all_nonlinear_rounds().clone()
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_challenges: &[F],
        _claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        let verifier_expr = self
            .expression
            .bind(sumcheck_challenges, transcript_reader)
            .map_err(VerificationError::ExpressionError)?;

        let verifier_layer = VerifierRegularLayer::new_raw(self.layer_id(), verifier_expr);
        Ok(verifier_layer)
    }

    /// Get the [PostSumcheckLayer] for a [CircuitRegularLayer], which represents the description of a fully bound expression.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        let nonlinear_round_indices = self.sumcheck_round_indices();
        // Filter the claim to get the values of the claim pertaining to the nonlinear rounds.
        let nonlinear_claim_points = claim_challenges
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                if nonlinear_round_indices.contains(&idx) {
                    Some(*point)
                } else {
                    None
                }
            })
            .collect_vec();
        assert_eq!(round_challenges.len(), nonlinear_claim_points.len());

        // Compute beta over these and the sumcheck challenges.
        let fully_bound_beta =
            BetaValues::compute_beta_over_two_challenges(round_challenges, &nonlinear_claim_points);

        // Compute the fully bound challenges, which include those pre-fixed for linear rounds
        // and the sumcheck rounds.
        let mut nonlinear_round_index_counter = 0;
        let all_bound_challenges = (0..claim_challenges.len())
            .map(|idx| {
                if nonlinear_round_indices.contains(&idx) {
                    let chal = round_challenges[nonlinear_round_index_counter];
                    nonlinear_round_index_counter += 1;
                    chal
                } else {
                    claim_challenges[idx]
                }
            })
            .collect_vec();

        assert_eq!(nonlinear_round_index_counter, nonlinear_round_indices.len());

        self.expression
            .get_post_sumcheck_layer(fully_bound_beta, &all_bound_challenges)
    }

    fn max_degree(&self) -> usize {
        self.expression.get_max_degree() + 1
    }

    fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>> {
        self.expression.get_circuit_mles()
    }

    fn convert_into_prover_layer(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let prover_expr = self.expression.into_prover_expression(circuit_map);
        let regular_layer = RegularLayer::new_raw(self.layer_id(), prover_expr);
        regular_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.expression.index_mle_indices(start_index);
    }
}

impl<F: Field> VerifierLayer<F> for VerifierRegularLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.id
    }
}

impl<F: Field> RegularLayer<F> {
    /// Initialize all necessary information in order to start sumcheck within a
    /// layer of GKR. This includes pre-fixing all of the rounds within the
    /// layer which are linear, and then appropriately initializing the
    /// necessary beta values over the nonlinear rounds.
    fn start_sumcheck(&mut self, claim: &Claim<F>) -> Result<(), LayerError> {
        println!("Starting sumcheck for layer: {:?}", self.id);
        let claim_point = claim.get_point();

        // Grab and index the expression.
        let expression = &mut self.expression;
        let expression_num_indices = expression.index_mle_indices(0);

        let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
        let expression_linear_indices = expression.get_all_linear_rounds();
        debug_assert_eq!(
            expression_num_indices,
            expression_nonlinear_indices.len() + expression_linear_indices.len()
        );

        // For each of the linear indices in the expression, we can fix the
        // variable at that index for the expression, so that now the only
        // unbound indices are the nonlinear indices.
        expression_linear_indices
            .into_iter()
            .sorted()
            .for_each(|round_idx| {
                expression.fix_variable_at_index(round_idx, claim_point[round_idx]);
            });

        // We need the beta values over the nonlinear indices of the
        // expression, so we grab the claim points that are over these
        // nonlinear indices and then initialize the betavalues struct over
        // them.
        let betavec = expression_nonlinear_indices
            .iter()
            .map(|idx| (*idx, claim_point[*idx]))
            .collect_vec();
        let newbeta = BetaValues::new(betavec);
        self.beta_vals = Some(newbeta);

        // Store the nonlinear rounds of the expression within the layer so
        // that we know these are the rounds we perform sumcheck over.
        self.nonlinear_rounds = Some(expression_nonlinear_indices);

        Ok(())
    }

    /// Performs a round of the sumcheck protocol on this Layer.
    fn prove_nonlinear_round(
        &mut self,
        transcript_writer: &mut impl ProverTranscript<F>,
        round_index: usize,
    ) -> Result<(), LayerError> {
        println!("Proving round: {round_index}");

        // Grabs the degree of univariate polynomial we are sending over.
        let degree = get_round_degree(&self.expression, round_index);

        // Compute the sumcheck message for this round.
        let prover_sumcheck_message = compute_sumcheck_message_beta_cascade(
            &self.expression,
            round_index,
            degree,
            self.beta_vals.as_ref().unwrap(),
        )?
        .0;

        transcript_writer.append_elements("Sumcheck message", &prover_sumcheck_message);

        let challenge = transcript_writer.get_challenge("Sumcheck challenge");

        self.expression.fix_variable(round_index, challenge);

        self.beta_vals
            .as_mut()
            .unwrap()
            .beta_update(round_index, challenge);

        Ok(())
    }

    /// Traverse the fully-bound `self.expression` and append all MLE values
    /// to the trascript.
    pub fn append_leaf_mles_to_transcript(
        &self,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        let mut observer_fn = |expr_node: &ExpressionNode<F, ProverExpr>,
                               mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec|
         -> Result<(), ()> {
            match expr_node {
                ExpressionNode::Mle(mle_vec_index) => {
                    let mle: &DenseMle<F> = &mle_vec[mle_vec_index.index()];
                    let val = mle.current_mle.value();
                    transcript_writer.append("Leaf MLE value", val);
                    Ok(())
                }
                ExpressionNode::Product(mle_vec_indices) => {
                    for mle_vec_index in mle_vec_indices {
                        let mle = &mle_vec[mle_vec_index.index()];
                        let eval = mle.current_mle.value();
                        transcript_writer.append("Product MLE value", eval);
                    }
                    Ok(())
                }
                ExpressionNode::Constant(_)
                | ExpressionNode::Scaled(_, _)
                | ExpressionNode::Sum(_, _)
                | ExpressionNode::Negated(_)
                | ExpressionNode::Selector(_, _, _) => Ok(()),
            }
        };

        let _ = self.expression.traverse(&mut observer_fn);

        Ok(())
    }

    ///Gets the expression that this layer is proving
    pub fn expression(&self) -> &Expression<F, ProverExpr> {
        &self.expression
    }

    /// Creates a new `RegularLayer` from an `Expression` and a `LayerId`
    ///
    /// The `Expression` is the relationship this `Layer` proves
    /// and the `LayerId` is the location of this `Layer` in the overall circuit
    pub fn new_raw(id: LayerId, expression: Expression<F, ProverExpr>) -> Self {
        RegularLayer {
            id,
            expression,
            nonlinear_rounds: None,
            beta_vals: None,
        }
    }

    pub(crate) fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        Self {
            id,
            expression: builder.build_expression(),
            nonlinear_rounds: None,
            beta_vals: None,
        }
    }
}

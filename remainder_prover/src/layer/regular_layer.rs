//! The implementation of `RegularLayer`

use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::{transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter}, FieldExt};
use serde::{Deserialize, Serialize};
use tracing::info;
use rayon::prelude::{ParallelIterator, IntoParallelIterator};

use crate::{layer::{combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs}, Layer, layer_builder::LayerBuilder, LayerError, LayerId, VerificationError}, claims::{wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals}, Claim, ClaimError, YieldClaim}, expression::{generic_expr::{Expression, ExpressionNode, ExpressionType}, prover_expr::ProverExpr}, mle::{betavalues::BetaValues, mle_enum::MleEnum, MleRef}, prover::SumcheckProof, sumcheck::{compute_sumcheck_message_beta_cascade, evaluate_at_a_point, get_round_degree, Evals}};

/// The most common implementation of `Layer`
/// 
/// A `Layer` made up of a structured polynomial relationship between MLEs of previous Layers
/// 
/// Proofs are generated with the Sumcheck protocol
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct RegularLayer<F: FieldExt> {
    id: LayerId,
    pub(crate) expression: Expression<F, ProverExpr>,
    /// store the nonlinear rounds in a GKR layer so that these are the only rounds we produce sumcheck proofs over.
    nonlinear_rounds: Option<Vec<usize>>,
    /// store the beta values associated with an expression.
    beta_vals: Option<BetaValues<F>>,
}

impl<F: FieldExt> Layer<F> for RegularLayer<F> {
    type Proof = Option<SumcheckProof<F>>;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<Option<SumcheckProof<F>>, LayerError> {
        let val = claim.get_result();

        // --- Initialize tables and compute prover message for first round of sumcheck ---
        let first_sumcheck_message = self.start_sumcheck(claim)?;

        let nonlinear_rounds = self.nonlinear_rounds.clone().unwrap();
        if nonlinear_rounds.is_empty() {
            return Ok(None);
        }

        info!("Proving GKR Layer");

        debug_assert_eq!(first_sumcheck_message[0] + first_sumcheck_message[1], val);

        // --- Add prover message to the FS transcript ---
        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_sumcheck_message);

        // Grabs all of the sumcheck messages from all of the rounds within this layer.
        //
        // Note that the sumcheck messages are g_1(x), ..., g_n(x) for an expression with
        // n iterated variables, where g_i(x) = \sum_{b_{i + 1}, ..., b_n} g(r_1, ..., r_{i - 1}, r_i, b_{i + 1}, ..., b_n)
        // and we always give the evals g_i(0), g_i(1), ..., g_i(d - 1) where `d` is the degree of the ith variable.
        //
        // Additionally, each of the `r_i`s is sampled from the FS transcript and the prover messages
        // (i.e. all of the g_i's) are added to the transcript each time.
        //
        // we only have sumcheck messages over the nonlinear rounds, so we iterate just through the nonlinear rounds
        // and compute each of the sumcheck messages since each of the linear rounds are already bound.
        let all_prover_sumcheck_messages: Vec<Vec<F>> = std::iter::once(Ok(first_sumcheck_message))
            .chain((nonlinear_rounds.iter().skip(1)).map(|round_index| {
                // --- Verifier samples a random challenge \in \mathbb{F} to send to prover ---
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");

                // --- Prover uses that random challenge to compute the next sumcheck message ---
                // --- We then add the prover message to FS transcript ---
                let prover_sumcheck_message =
                    self.prove_nonlinear_round(*round_index, challenge)?;
                transcript_writer.append_elements("Sumcheck evaluations", &prover_sumcheck_message);
                Ok::<_, LayerError>(prover_sumcheck_message)
            }))
            .collect::<Result<_, _>>()?;

        // --- For the final round, we need to check that g(r_1, ..., r_n) = g_n(r_n) ---
        // --- Thus we sample r_n and bind b_n to it (via `fix_variable` below) ---
        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge");

        let last_idx = nonlinear_rounds[nonlinear_rounds.len() - 1];

        self.expression.fix_variable(last_idx, final_chal);
        if let Some(beta) = self.beta_vals.as_mut() {
            beta.beta_update(last_idx, final_chal)
        }
        Ok(Some(all_prover_sumcheck_messages.into()))
    }

    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_prover_messages: Self::Proof,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError> {
        // if there are no sumcheck prover messages, this means this was an entirely linear layer. therefore
        // we can skip the verification for this layer.
        if sumcheck_prover_messages.is_none() {
            return Ok(());
        }
        let sumcheck_prover_messages = sumcheck_prover_messages.unwrap();
        // --- Keeps track of challenges u_1, ..., u_n to be bound ---
        let mut challenges = vec![];
        let sumcheck_prover_messages: Vec<Vec<F>> = sumcheck_prover_messages.0;

        // --- First verify that g_1(0) + g_1(1) = \sum_{b_1, ..., b_n} g(b_1, ..., b_n) ---
        // (i.e. the first verification step of sumcheck)

        // TODO(Makis): Retrieve `num_prev_evals` directly from the transcript.
        let num_prev_evals = sumcheck_prover_messages[0].len();
        let mut prev_evals = transcript_reader
            .consume_elements("Initial Sumcheck evaluations", num_prev_evals)?;

        if prev_evals[0] + prev_evals[1] != claim.get_result() {
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        // --- For round 1 < i < n, perform the check ---
        // g_{i - 1}(r_i) = g_i(0) + g_i(1)
        // TODO(Makis): Retrieve `evals.len()`s directly from the transcript.
        for num_curr_evals in sumcheck_prover_messages
            .into_iter()
            .skip(1)
            .map(|evals| evals.len())
        {
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")?;

            let prev_at_r =
                evaluate_at_a_point(&prev_evals, challenge)?;

            let curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", num_curr_evals)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            prev_evals = curr_evals;
            challenges.push(challenge);
        }

        // --- In the final round, we check that g(r_1, ..., r_n) = g_n(r_n) ---
        // Here, we first sample r_n.
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")?;
        challenges.push(final_chal);

        // --- This automatically asserts that the expression is fully bound and simply ---
        // --- attempts to combine/collect the expression evaluated at the (already bound) challenge coords ---
        let expr_evaluated_at_challenge_coord = self
            .expression
            .clone()
            .transform_to_verifier_expression()
            .unwrap()
            .gather_combine_all_evals()
            .unwrap();

        // --- Simply computes \beta((g_1, ..., g_n), (u_1, ..., u_n)) for claim coords (g_1, ..., g_n) and ---
        // --- bound challenges (u_1, ..., u_n) ---
        let expr_nonlinear_indices = self.nonlinear_rounds.as_ref().unwrap();
        let claim_nonlinear_vals: Vec<F> = expr_nonlinear_indices
            .iter()
            .map(|idx| (claim.get_point()[*idx]))
            .collect();
        let beta_fn_evaluated_at_challenge_point =
            BetaValues::compute_beta_over_two_challenges(&claim_nonlinear_vals, &challenges);

        // --- The actual value should just be the product of the two ---
        let mle_evaluated_at_challenge_coord =
            expr_evaluated_at_challenge_coord * beta_fn_evaluated_at_challenge_point;

        // --- Computing g_n(r_n) ---
        let g_n_evaluated_at_r_n =
            evaluate_at_a_point(&prev_evals, final_chal)?;

        // --- Checking the two against one another ---
        if mle_evaluated_at_challenge_coord != g_n_evaluated_at_r_n {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    fn id(&self) -> &LayerId {
        &self.id
    }
}

impl<F: FieldExt> RegularLayer<F> {
    /// initialize all necessary information in order to start sumcheck within a layer of GKR. this includes
    /// pre-fixing all of the rounds within the layer which are linear,
    /// and then appropriately initializing the necessary beta values over the nonlinear rounds.
    ///
    /// this function returns the evaluations over the first round of sumcheck, which is over the first
    /// nonlinear index in the expression we are performing GKR over. if there are no nonlinear rounds
    /// it returns an empty vector because there is no sumcheck proof for this layer.
    fn start_sumcheck(&mut self, claim: Claim<F>) -> Result<Vec<F>, LayerError> {
        let first_nonlinear_round = {
            let expression = &mut self.expression;
            let _expression_num_indices = expression.index_mle_indices(0);
            let expression_nonlinear_indices = expression.get_all_nonlinear_rounds();
            let expression_linear_indices = expression.get_all_linear_rounds();

            let claim_point = claim.get_point();

            // for each of the linear indices in the expression, we can fix the variable at that index for
            // the expression, so that now the only unbound indices are the nonlinear indices.
            expression_linear_indices
                .iter()
                .sorted()
                .for_each(|round_idx| {
                    expression.fix_variable_at_index(*round_idx, claim_point[*round_idx]);
                });

            // now we only need the beta values over the nonlinear indices of the expression, so we grab
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
            let first_nonlinear_round =  expression_nonlinear_indices.get(0).cloned();
            self.nonlinear_rounds = Some(expression_nonlinear_indices);

            // if there are no nonlinear indices in the expression we can return an empty vector early.
            if first_nonlinear_round.is_none() {
                return Ok(vec![]);
            }
            // otherwise we know the first nonlinear round is the first value here because these are sorted.
            first_nonlinear_round.unwrap()
        };

        // --- Grabs the expression/beta table/variable degree for the first round and executes the sumcheck prover for the first round ---
        let expression = &mut self.expression;
        let new_beta = &mut self.beta_vals;
        let degree = get_round_degree(expression, first_nonlinear_round);

        // we compute the first sumcheck message which is the evaluations when the independent variable
        // is the first nonlinear index in the expression.
        let first_round_message = compute_sumcheck_message_beta_cascade(
            expression,
            first_nonlinear_round,
            degree,
            new_beta.as_ref().unwrap(),
        );

        // let Evals(out) = first_round_sumcheck_message;
        let Evals(out) = first_round_message.unwrap();

        Ok(out)
    }

    /// Computes a round of the sumcheck protocol on this Layer
    fn prove_nonlinear_round(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<Vec<F>, LayerError> {
        // --- Grabs the expression/beta table and updates them with the new challenge ---
        let expression = &mut self.expression;
        let newbeta = &mut self.beta_vals;
        expression.fix_variable(round_index - 1, challenge);
        newbeta
            .as_mut()
            .unwrap()
            .beta_update(round_index - 1, challenge);

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expression, round_index);

        let prover_sumcheck_message = compute_sumcheck_message_beta_cascade(
            expression,
            round_index,
            degree,
            newbeta.as_ref().unwrap(),
        )
        .unwrap();

        Ok(prover_sumcheck_message.0)
    }

    ///Gets the expression that this layer is proving
    pub fn expression(&self) -> &Expression<F, ProverExpr> {
        &self.expression
    }

    pub(crate) fn new_raw(id: LayerId, expression: Expression<F, ProverExpr>) -> Self {
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

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for RegularLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        // First off, parse the expression that is associated with the layer...
        // Next, get to the actual claims that are generated by each expression and grab them
        // Return basically a list of (usize, Claim)
        let layerwise_expr = &self.expression;

        // --- Define how to parse the expression tree ---
        // - Basically we just want to go down it and pass up claims
        // - We can only add a new claim if we see an MLE with all its indices bound

        let mut claims: Vec<ClaimMle<F>> = Vec::new();

        let mut observer_fn =
            |exp: &ExpressionNode<F, ProverExpr>,
             mle_vec: &<ProverExpr as ExpressionType<F>>::MleVec| {
                match exp {
                    ExpressionNode::Mle(mle_vec_idx) => {
                        let mle_ref = mle_vec_idx.get_mle(mle_vec);

                        let fixed_mle_indices = mle_ref.mle_indices.iter().map(|index| index.val().ok_or(ClaimError::MleRefMleError)).collect::<Result<Vec<_>, _>>()?;

                        // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                        let mle_layer_id = mle_ref.get_layer_id();

                        // --- Grab the actual value that the claim is supposed to evaluate to ---
                        if mle_ref.bookkeeping_table().len() != 1 {
                            return Err(ClaimError::MleRefMleError);
                        }
                        let claimed_value = mle_ref.bookkeeping_table()[0];

                        // --- Construct the claim ---
                        let claim: ClaimMle<F> = ClaimMle::new(
                            fixed_mle_indices,
                            claimed_value,
                            Some(*self.id()),
                            Some(mle_layer_id),
                            Some(MleEnum::Dense(mle_ref.clone())),
                        );

                        // --- Push it into the list of claims ---
                        claims.push(claim);
                    }
                    ExpressionNode::Product(mle_vec_indices) => {
                        for mle_vec_index in mle_vec_indices {
                            let mle_ref = mle_vec_index.get_mle(mle_vec);

                            let fixed_mle_indices = mle_ref.mle_indices.iter().map(|index| index.val().ok_or(ClaimError::MleRefMleError)).collect::<Result<Vec<_>, _>>()?;

                            // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                            let mle_layer_id = mle_ref.get_layer_id();

                            // --- Grab the actual value that the claim is supposed to evaluate to ---

                            if mle_ref.bookkeeping_table().len() != 1 {
                                return Err(ClaimError::MleRefMleError);
                            }
                            let claimed_value = mle_ref.bookkeeping_table()[0];

                            // --- Construct the claim ---
                            // need to populate the claim with the mle ref we are grabbing the claim from
                            let claim: ClaimMle<F> = ClaimMle::new(
                                fixed_mle_indices,
                                claimed_value,
                                Some(*self.id()),
                                Some(mle_layer_id),
                                Some(MleEnum::Dense(mle_ref.clone())),
                            );

                            // --- Push it into the list of claims ---
                            claims.push(claim);
                        }
                    }
                    _ => {}
                }
                Ok(())
            };

        // --- Apply the observer function from above onto the expression ---
        layerwise_expr
            .traverse(&mut observer_fn)?;

        Ok(claims)
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for RegularLayer<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claim_mle_refs: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);

        let mut claim_mle_refs = claim_mle_refs;

        if common_idx.is_some() {
            pre_fix_mle_refs(&mut claim_mle_refs, &claim_vecs[0], common_idx.unwrap());
        }

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                let wlx_eval_on_mle_ref =
                    combine_mle_refs_with_aggregate(&claim_mle_refs, &new_chal);
                wlx_eval_on_mle_ref.unwrap()
            })
            .collect();

        // concat this with the first k evaluations from the claims to
        // get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        Ok(wlx_evals)
    }
}

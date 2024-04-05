//! A layer is a combination of multiple MLEs with an expression

pub mod batched;
pub mod combine_mle_refs;
pub mod empty_layer;
pub mod layer_enum;
pub mod simple_builders;
// mod gkr_layer;

use std::{fmt::Debug, marker::PhantomData};

use ark_std::cfg_into_iter;
use itertools::repeat_n;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::Value;

use crate::{
    claims::{wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals, ENABLE_PRE_FIX, ENABLE_RAW_MLE}, Claim, ClaimError, YieldClaim},
    expression::{
        expr_errors::ExpressionError,
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    mle::{
        beta::{compute_beta_over_two_challenges, BetaError, BetaTable},
        dense::DenseMleRef,
        mle_enum::MleEnum,
        MleIndex, MleRef,
    },
    prover::{SumcheckProof, ENABLE_OPTIMIZATION},
    sumcheck::{
        compute_sumcheck_message, evaluate_at_a_point, get_round_degree, Evals, InterpError,
    },
};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptReaderError, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

use self::{
    combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs},
    layer_enum::LayerEnum,
};

use core::cmp::Ordering;

use log::{debug, info};

#[derive(Error, Debug, Clone)]
/// Errors to do with working with a Layer
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    /// Layer isn't ready to prove
    LayerNotReady,
    #[error("Error with underlying expression: {0}")]
    /// Error with underlying expression: {0}
    ExpressionError(ExpressionError),
    #[error("Error with aggregating curr layer")]
    /// Error with aggregating curr layer
    AggregationError,
    #[error("Error with getting Claim: {0}")]
    /// Error with getting Claim
    ClaimError(ClaimError),
    #[error("Error with verifying layer: {0}")]
    /// Error with verifying layer
    VerificationError(VerificationError),
    #[error("Beta Error: {0}")]
    /// Beta Error
    BetaError(BetaError),
    #[error("InterpError: {0}")]
    /// InterpError
    InterpError(InterpError),
    #[error("Transcript Error: {0}")]
    /// Transcript Error
    TranscriptError(TranscriptReaderError),
}

#[derive(Error, Debug, Clone)]
/// Errors to do with verifying a Layer
pub enum VerificationError {
    #[error("The sum of the first evaluations do not equal the claim")]
    /// The sum of the first evaluations do not equal the claim
    SumcheckStartFailed,
    #[error("The sum of the current rounds evaluations do not equal the previous round at a random point")]
    /// The sum of the current rounds evaluations do not equal the previous round at a random point
    SumcheckFailed,
    #[error("The final rounds evaluations at r do not equal the oracle query")]
    /// The final rounds evaluations at r do not equal the oracle query
    FinalSumcheckFailed,
    #[error("The Oracle query does not match the final claim")]
    /// The Oracle query does not match the final claim
    GKRClaimCheckFailed,
    #[error(
        "The Challenges generated during sumcheck don't match the claims in the given expression"
    )]
    ///The Challenges generated during sumcheck don't match the claims in the given expression
    ChallengeCheckFailed,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
///  The location of a layer within the GKR circuit
pub enum LayerId {
    /// A random mle input layer
    RandomInput(usize),
    /// An Mle located in the input layer
    Input(usize),
    /// A layer within the GKR protocol, indexed by it's layer id
    Layer(usize),
    /// An MLE located in the output layer.
    Output(usize),
}

impl Ord for LayerId {
    fn cmp(&self, layer2: &LayerId) -> Ordering {
        match (self, layer2) {
            (LayerId::RandomInput(id1), LayerId::RandomInput(id2)) => id1.cmp(&id2),
            (LayerId::RandomInput(id1), _) => Ordering::Less,
            (LayerId::Input(id1), LayerId::Input(id2)) => id1.cmp(&id2),
            (LayerId::Input(id1), _) => Ordering::Less,
            (LayerId::Layer(id1), LayerId::Input(id2)) => Ordering::Greater,
            (LayerId::Layer(id1), LayerId::Layer(id2)) => id1.cmp(&id2),
            (LayerId::Layer(id1), _) => Ordering::Less,
            (LayerId::Output(id1), LayerId::Output(id2)) => id1.cmp(&id2),
            (LayerId::Output(id1), _) => Ordering::Greater,
        }
    }
}

impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    type Proof: Debug + Serialize + for<'a> Deserialize<'a>;

    /// Creates a sumcheck proof for this Layer
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::Proof, LayerError>;

    ///  Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        proof: Self::Proof,
        transcript: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError>;

    /// Get the claims that this layer makes on other layers
    // fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError>;

    /// Gets this layers id
    fn id(&self) -> &LayerId;
}

/// Default Layer abstraction
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")] 
pub struct GKRLayer<F: FieldExt> {
    id: LayerId,
    pub(crate) expression: Expression<F, ProverExpr>,
    beta: Option<BetaTable<F>>,
}

impl<F: FieldExt> GKRLayer<F> {
    /// Ingest a claim, initialize beta tables, and do any other
    /// bookkeeping that needs to be done before the sumcheck starts
    fn start_sumcheck(&mut self, claim: Claim<F>) -> Result<(Vec<F>, usize), LayerError> {
        // --- `max_round` is total number of rounds of sumcheck which need to be performed ---
        // --- `beta` is the beta table itself, initialized with the challenge coordinate held within `claim` ---
        let (max_round, beta) = {
            let (expression, _) = self.mut_expression_and_beta();

            let mut beta =
                BetaTable::new(claim.get_point().clone()).map_err(LayerError::BetaError)?;

            let expression_num_indices = expression.index_mle_indices(0);
            let beta_table_num_indices = beta.table.index_mle_indices(0);

            // --- This should always be equivalent to the number of indices within the beta table ---
            let max_round = std::cmp::max(expression_num_indices, beta_table_num_indices);
            (max_round, beta)
        };

        // --- Sets the beta table for the current layer we are sumchecking over ---
        self.set_beta(beta);

        // --- Grabs the expression/beta table/variable degree for the first round and executes the sumcheck prover for the first round ---
        let (expression, beta) = self.mut_expression_and_beta();
        let beta = beta.as_ref().unwrap();
        let degree = get_round_degree(expression, 0);
        let first_round_sumcheck_message = compute_sumcheck_message(expression, 0, degree, beta)
            .map_err(LayerError::ExpressionError)?;

        let Evals(out) = first_round_sumcheck_message;

        Ok((out, max_round))
    }

    /// Computes a round of the sumcheck protocol on this Layer
    fn prove_round(&mut self, round_index: usize, challenge: F) -> Result<Vec<F>, LayerError> {
        // --- Grabs the expression/beta table and updates them with the new challenge ---
        let (expression, beta) = self.mut_expression_and_beta();
        let beta = beta.as_mut().ok_or(LayerError::LayerNotReady)?;
        //dbg!(&expression);
        expression.fix_variable(round_index - 1, challenge);
        beta.beta_update(round_index - 1, challenge)
            .map_err(LayerError::BetaError)?;

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expression, round_index);

        let prover_sumcheck_message =
            compute_sumcheck_message(expression, round_index, degree, beta)
                .map_err(LayerError::ExpressionError)?;

        Ok(prover_sumcheck_message.0)
    }

    fn mut_expression_and_beta(
        &mut self,
    ) -> (&mut Expression<F, ProverExpr>, &mut Option<BetaTable<F>>) {
        (&mut self.expression, &mut self.beta)
    }

    fn set_beta(&mut self, beta: BetaTable<F>) {
        self.beta = Some(beta);
    }

    ///Gets the expression that this layer is proving
    pub fn expression(&self) -> &Expression<F, ProverExpr> {
        &self.expression
    }

    pub(crate) fn new_raw(id: LayerId, expression: Expression<F, ProverExpr>) -> Self {
        GKRLayer {
            id,
            expression,
            beta: None,
        }
    }

    pub(crate) fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        Self {
            id,
            expression: builder.build_expression(),
            beta: None,
        }
    }
}

impl<F: FieldExt> Layer<F> for GKRLayer<F> {
    type Proof = SumcheckProof<F>;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let val = claim.get_result().clone();

        // --- Initialize tables and compute prover message for first round of sumcheck ---
        let (first_sumcheck_message, num_sumcheck_rounds) = self.start_sumcheck(claim)?;
        if val != first_sumcheck_message[0] + first_sumcheck_message[1] {
            dbg!(&val);
            dbg!(first_sumcheck_message[0] + first_sumcheck_message[1]);
            dbg!(&self.expression);
        }

        info!("Proving GKR Layer");
        if first_sumcheck_message[0] + first_sumcheck_message[1] != val {
            debug!("HUGE PROBLEM");
        }
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
        let all_prover_sumcheck_messages: Vec<Vec<F>> = std::iter::once(Ok(first_sumcheck_message))
            .chain((1..num_sumcheck_rounds).map(|round_index| {
                // --- Verifier samples a random challenge \in \mathbb{F} to send to prover ---
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");

                // --- Prover uses that random challenge to compute the next sumcheck message ---
                // --- We then add the prover message to FS transcript ---
                let prover_sumcheck_message = self.prove_round(round_index, challenge)?;
                transcript_writer.append_elements("Sumcheck evaluations", &prover_sumcheck_message);
                Ok::<_, LayerError>(prover_sumcheck_message)
            }))
            .collect::<Result<_, _>>()?;

        // --- For the final round, we need to check that g(r_1, ..., r_n) = g_n(r_n) ---
        // --- Thus we sample r_n and bind b_n to it (via `fix_variable` below) ---
        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge");

        self.expression
            .fix_variable(num_sumcheck_rounds - 1, final_chal);
        self.beta
            .as_mut()
            .map(|beta| beta.beta_update(num_sumcheck_rounds - 1, final_chal));

        Ok(all_prover_sumcheck_messages.into())
    }

    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_prover_messages: Self::Proof,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError> {
        // --- Keeps track of challenges u_1, ..., u_n to be bound ---
        let mut challenges = vec![];
        let sumcheck_prover_messages: Vec<Vec<F>> = sumcheck_prover_messages.0;

        // --- First verify that g_1(0) + g_1(1) = \sum_{b_1, ..., b_n} g(b_1, ..., b_n) ---
        // (i.e. the first verification step of sumcheck)

        // TODO(Makis): Retrieve `num_prev_evals` directly from the transcript.
        let num_prev_evals = sumcheck_prover_messages[0].len();
        let mut prev_evals = transcript_reader
            .consume_elements("Initial Sumcheck evaluations", num_prev_evals)
            .map_err(|e| LayerError::TranscriptError(e))?;

        if prev_evals[0] + prev_evals[1] != claim.get_result() {
            debug!("I'm the PROBLEM");
            debug!("msg0 + msg1 =\n{:?}", prev_evals[0] + prev_evals[1]);
            debug!("rest =\n{:?}", claim.get_result());
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
                .get_challenge("Sumcheck challenge")
                .map_err(|e| LayerError::TranscriptError(e))?;

            let prev_at_r =
                evaluate_at_a_point(&prev_evals, challenge).map_err(LayerError::InterpError)?;

            let curr_evals = transcript_reader
                .consume_elements("Sumcheck evaluations", num_curr_evals)
                .map_err(|e| LayerError::TranscriptError(e))?;

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
            .get_challenge("Final Sumcheck challenge")
            .map_err(|e| LayerError::TranscriptError(e))?;
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
        let beta_fn_evaluated_at_challenge_point =
            compute_beta_over_two_challenges(claim.get_point(), &challenges);

        // --- The actual value should just be the product of the two ---
        let mle_evaluated_at_challenge_coord =
            expr_evaluated_at_challenge_coord * beta_fn_evaluated_at_challenge_point;

        // --- Computing g_n(r_n) ---
        let g_n_evaluated_at_r_n =
            evaluate_at_a_point(&prev_evals, final_chal).map_err(LayerError::InterpError)?;

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

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for GKRLayer<F> {
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

                        // --- First ensure that all the indices are fixed ---
                        let mle_indices = mle_ref.mle_indices();

                        // --- This is super jank ---
                        let mut fixed_mle_indices: Vec<F> = vec![];
                        for mle_idx in mle_indices {
                            if mle_idx.val().is_none() {
                                dbg!("We got a nothing");
                                dbg!(&mle_idx);
                                dbg!(&mle_indices);
                                dbg!(&mle_ref);
                            }
                            fixed_mle_indices
                                .push(mle_idx.val().ok_or(ClaimError::MleRefMleError)?);
                        }

                        // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                        let mle_layer_id = mle_ref.get_layer_id();

                        // --- Grab the actual value that the claim is supposed to evaluate to ---
                        if mle_ref.bookkeeping_table().len() != 1 {
                            dbg!(&mle_ref.current_mle);
                            return Err(ClaimError::MleRefMleError);
                        }
                        let claimed_value = mle_ref.bookkeeping_table()[0];

                        // --- Construct the claim ---
                        // println!("========\n I'm making a GKR layer claim for an MLE!!\n==========");
                        // println!("From: {:#?}, To: {:#?}", self.id().clone(), mle_layer_id);
                        let claim: ClaimMle<F> = ClaimMle::new(
                            fixed_mle_indices,
                            claimed_value,
                            Some(self.id().clone()),
                            Some(mle_layer_id),
                            Some(MleEnum::Dense(mle_ref.clone())),
                        );

                        // --- Push it into the list of claims ---
                        claims.push(claim);
                    }
                    ExpressionNode::Product(mle_vec_indices) => {
                        for mle_vec_index in mle_vec_indices {
                            let mle_ref = mle_vec_index.get_mle(mle_vec);

                            // --- First ensure that all the indices are fixed ---
                            let mle_indices = mle_ref.mle_indices();

                            // --- This is super jank ---
                            let mut fixed_mle_indices: Vec<F> = vec![];
                            for mle_idx in mle_indices {
                                fixed_mle_indices
                                    .push(mle_idx.val().ok_or(ClaimError::MleRefMleError)?);
                            }

                            // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                            let mle_layer_id = mle_ref.get_layer_id();

                            // --- Grab the actual value that the claim is supposed to evaluate to ---

                            if mle_ref.bookkeeping_table().len() != 1 {
                                dbg!(&mle_ref);
                                return Err(ClaimError::MleRefMleError);
                            }
                            let claimed_value = mle_ref.bookkeeping_table()[0];

                            // --- Construct the claim ---
                            // need to populate the claim with the mle ref we are grabbing the claim from
                            let claim: ClaimMle<F> = ClaimMle::new(
                                fixed_mle_indices,
                                claimed_value,
                                Some(self.id().clone()),
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
            .traverse(&mut observer_fn)
            .map_err(LayerError::ClaimError)?;

        Ok(claims)
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for GKRLayer<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claim_mle_refs: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        let mut expr = self.expression.clone();

        //fix variable hella times
        //evaluate expr on the mutated expr

        // get the number of evaluations
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);

        let mut claim_mle_refs = claim_mle_refs.clone();

        if ENABLE_PRE_FIX {
            if common_idx.is_some() {
                pre_fix_mle_refs(&mut claim_mle_refs, &claim_vecs[0], common_idx.unwrap());
            }
        }

        // TODO(Makis): This assert fails on `test_aggro_claim_4` and I'm not
        // sure if the test is wrong or if the assert is wrong!
        /*
        debug_assert!({
            claim_vecs.iter().zip(claimed_vals.iter()).map(|(point, val)| {
                let mut beta = BetaTable::new(point.to_vec()).unwrap();
                beta.table.index_mle_indices(0);
                let eval = compute_sumcheck_message(&mut expr.clone(), 0, degree, &beta).unwrap();
                let Evals(evals) = eval;
                let eval = evals[0] + evals[1];
                if eval == *val {
                    true
                } else {
                    dbg!(self.id());
                    dbg!(self.expression());
                    println!("Claim passed into compute_wlx is invalid! point is {:?} claimed val is {:?}, actual eval is {:?}", point, val, eval);
                    false
                }
            }).reduce(|acc, val| acc && val).unwrap()
        });
        */

        let mut degree = 0;
        if !ENABLE_RAW_MLE {
            expr.index_mle_indices(0);
            degree = get_round_degree(&expr, 0);
        }

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                if !ENABLE_RAW_MLE {
                    let mut beta = BetaTable::new(new_chal).unwrap();
                    beta.table.index_mle_indices(0);
                    let eval = compute_sumcheck_message(&expr, 0, degree, &beta).unwrap();
                    let Evals(evals) = eval;
                    evals[0] + evals[1]
                } else {
                    let wlx_eval_on_mle_ref =
                        combine_mle_refs_with_aggregate(&claim_mle_refs, &new_chal);
                    wlx_eval_on_mle_ref.unwrap()
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to
        // get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        Ok(wlx_evals)
    }
}

/// The builder type for a Layer
pub trait LayerBuilder<F: FieldExt> {
    /// The layer that makes claims on this layer in the GKR protocol. The next layer in the GKR protocol
    type Successor;

    /// Build the expression that will be sumchecked
    fn build_expression(&self) -> Expression<F, ProverExpr>;

    /// Generate the next layer
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor;

    /// Concatonate two layers together
    fn concat<Other: LayerBuilder<F>>(self, rhs: Other) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            padding: Padding::None,
            _marker: PhantomData,
        }
    }

    ///Concatonate two layers together with some padding
    fn concat_with_padding<Other: LayerBuilder<F>>(
        self,
        rhs: Other,
        padding: Padding,
    ) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            padding,
            _marker: PhantomData,
        }
    }
}

/// Creates a simple layer from an mle, with closures for defining how the mle turns into an expression and a next layer
pub fn from_mle<
    F: FieldExt,
    M,
    EFn: Fn(&M) -> Expression<F, ProverExpr>,
    S,
    LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
>(
    mle: M,
    expression_builder: EFn,
    layer_builder: LFn,
) -> SimpleLayer<M, EFn, LFn> {
    SimpleLayer {
        mle,
        expression_builder,
        layer_builder,
    }
}

pub enum Padding {
    Right(usize),
    Left(usize),
    None,
}

/// The layerbuilder that represents two layers concatonated together
pub struct ConcatLayer<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> {
    first: A,
    second: B,
    padding: Padding,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> LayerBuilder<F> for ConcatLayer<F, A, B> {
    type Successor = (A::Successor, B::Successor);

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let first = self.first.build_expression();
        let second = self.second.build_expression();

        // return first.concat_expr(second);

        let zero_expression: Expression<F, ProverExpr> = Expression::constant(F::zero());

        let first_padded = if let Padding::Left(padding) = self.padding {
            let mut left = first;
            for _ in 0..padding {
                left = zero_expression.clone().concat_expr(left);
            }
            left
        } else {
            first
        };

        let second_padded = if let Padding::Right(padding) = self.padding {
            let mut right = second;
            for _ in 0..padding {
                right = zero_expression.clone().concat_expr(right);
            }
            right
        } else {
            second
        };

        first_padded.concat_expr(second_padded)
        // Expression::Selector(MleIndex::Iterated, Box::new(first_padded), Box::new(second_padded))
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let first_padding = if let Padding::Left(padding) = self.padding {
            repeat_n(MleIndex::Fixed(false), padding)
        } else {
            repeat_n(MleIndex::Fixed(false), 0)
        };
        let second_padding = if let Padding::Right(padding) = self.padding {
            repeat_n(MleIndex::Fixed(false), padding)
        } else {
            repeat_n(MleIndex::Fixed(false), 0)
        };
        (
            self.first.next_layer(
                id,
                Some(
                    prefix_bits
                        .clone()
                        .into_iter()
                        .flatten()
                        .chain(first_padding)
                        .chain(std::iter::once(MleIndex::Fixed(true)))
                        .collect(),
                ),
            ),
            self.second.next_layer(
                id,
                Some(
                    prefix_bits
                        .into_iter()
                        .flatten()
                        .chain(second_padding)
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .collect(),
                ),
            ),
        )
    }
}

/// A simple layer defined ad-hoc with two closures
pub struct SimpleLayer<M, EFn, LFn> {
    mle: M,
    expression_builder: EFn,
    layer_builder: LFn,
}

impl<
        F: FieldExt,
        M,
        EFn: Fn(&M) -> Expression<F, ProverExpr>,
        S,
        LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
    > LayerBuilder<F> for SimpleLayer<M, EFn, LFn>
{
    type Successor = S;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        (self.expression_builder)(&self.mle)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        (self.layer_builder)(&self.mle, id, prefix_bits)
    }
}

#[cfg(test)]
mod tests {

    // #[test]
    // fn build_simple_layer() {
    //     let mut rng = test_rng();
    //     let mle1 =
    //         DenseMle::<Fr, Fr>::new(vec![Fr::from(2), Fr::from(3), Fr::from(6), Fr::from(7)]);
    //     let mle2 =
    //         DenseMle::<Fr, Fr>::new(vec![Fr::from(3), Fr::from(1), Fr::from(9), Fr::from(2)]);

    //     let builder = from_mle(
    //         (mle1, mle2),
    //         |(mle1, mle2)| {
    //             Expression::mle(mle1.mle_ref()) + Expression::mle(mle2.mle_ref())
    //         },
    //         |(mle1, mle2), _, _: Option<Vec<MleIndex<Fr>>>| {
    //             mle1.clone()
    //                 .into_iter()
    //                 .zip(mle2.clone().into_iter())
    //                 .map(|(first, second)| first + second)
    //                 .collect::<DenseMle<_, _>>()
    //         },
    //     );

    //     let next: DenseMle<Fr, Fr> = builder.next_layer(LayerId::Layer(0), None);

    //     let mut layer = GKRLayer::<_, PoseidonTranscript<Fr>>::new(builder, LayerId::Layer(0));

    //     let sum = dummy_sumcheck(&mut layer.expression, &mut rng, todo!());
    //     verify_sumcheck_messages(sum, layer.expression, todo!(), &mut OsRng).unwrap();
    // }
}

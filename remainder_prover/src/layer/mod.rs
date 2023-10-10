//! A layer is a combination of multiple MLEs with an expression

pub mod batched;
pub mod claims;
pub mod empty_layer;
pub mod layer_enum;
pub mod combine_mle_refs;
// mod gkr_layer;

use std::marker::PhantomData;

use ark_std::cfg_into_iter;
use itertools::repeat_n;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    expression::{gather_combine_all_evals, Expression, ExpressionError, ExpressionStandard},
    mle::{
        beta::{compute_beta_over_two_challenges, BetaError, BetaTable},
        MleIndex, MleRef, dense::DenseMleRef, mle_enum::MleEnum,
    },
    prover::{SumcheckProof, ENABLE_OPTIMIZATION},
    sumcheck::{
        compute_sumcheck_message, evaluate_at_a_point, get_round_degree, Evals, InterpError,
    }, 
};
use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};

use self::{
    claims::{Claim, ClaimError},
    layer_enum::LayerEnum, combine_mle_refs::combine_mle_refs_with_aggregate,
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
    TranscriptError(TranscriptError),
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

/// A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    /// The transcript that this layer uses
    type Transcript: Transcript<F>;

    /// Creates a sumcheck proof for this Layer
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError>;

    ///  Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_rounds: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError>;

    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError>;

    /// Gets this layers id
    fn id(&self) -> &LayerId;

    ///Get W(l(x)) evaluations
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mle_refs: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError>;

    /// Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self
    where
        Self: Sized;

    fn get_enum(self) -> LayerEnum<F, Self::Transcript>;
}

/// Default Layer abstraction
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GKRLayer<F, Tr> {
    id: LayerId,
    pub(crate) expression: ExpressionStandard<F>,
    beta: Option<BetaTable<F>>,
    #[serde(skip)]
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> GKRLayer<F, Tr> {
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
            // dbg!(&expression_num_indices);
            // dbg!(&beta_table_num_indices);
            // dbg!(&expression);

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
    ) -> (&mut ExpressionStandard<F>, &mut Option<BetaTable<F>>) {
        (&mut self.expression, &mut self.beta)
    }

    fn set_beta(&mut self, beta: BetaTable<F>) {
        self.beta = Some(beta);
    }

    ///Gets the expression that this layer is proving
    pub fn expression(&self) -> &ExpressionStandard<F> {
        &self.expression
    }

    pub(crate) fn new_raw(id: LayerId, expression: ExpressionStandard<F>) -> Self {
        GKRLayer {
            id,
            expression,
            beta: None,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, Tr: Transcript<F>> Layer<F> for GKRLayer<F, Tr> {
    type Transcript = Tr;
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self {
        Self {
            id,
            expression: builder.build_expression(),
            beta: None,
            _marker: PhantomData,
        }
    }

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<SumcheckProof<F>, LayerError> {
        let val = claim.get_result().clone();

        // --- Initialize tables and compute prover message for first round of sumcheck ---
        let (first_sumcheck_message, num_sumcheck_rounds) = self.start_sumcheck(claim)?;

        if val != first_sumcheck_message[0] + first_sumcheck_message[1] {
            dbg!(&self.expression);
        }

        info!("Proving GKR Layer");
        if first_sumcheck_message[0] + first_sumcheck_message[1] != val {
            debug!("HUGE PROBLEM");
        }
        debug_assert_eq!(first_sumcheck_message[0] + first_sumcheck_message[1], val);

        // --- Add prover message to the FS transcript ---
        transcript
            .append_field_elements("Initial Sumcheck evaluations", &first_sumcheck_message)
            .unwrap();

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
                let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();

                // --- Prover uses that random challenge to compute the next sumcheck message ---
                // --- We then add the prover message to FS transcript ---
                let prover_sumcheck_message = self.prove_round(round_index, challenge)?;
                transcript
                    .append_field_elements("Sumcheck evaluations", &prover_sumcheck_message)
                    .unwrap();
                Ok::<_, LayerError>(prover_sumcheck_message)
            }))
            .collect::<Result<_, _>>()?;

        // --- For the final round, we need to check that g(r_1, ..., r_n) = g_n(r_n) ---
        // --- Thus we sample r_n and bind b_n to it (via `fix_variable` below) ---
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();

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
        sumcheck_prover_messages: Vec<Vec<F>>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), LayerError> {
        // --- Keeps track of challenges u_1, ..., u_n to be bound ---
        let mut challenges = vec![];

        // --- First verify that g_1(0) + g_1(1) = \sum_{b_1, ..., b_n} g(b_1, ..., b_n) ---
        // (i.e. the first verification step of sumcheck)
        let mut prev_evals = &sumcheck_prover_messages[0];

        if prev_evals[0] + prev_evals[1] != claim.get_result() {
            debug!("I'm the PROBLEM");
            debug!("msg0 + msg1 =\n{:?}", prev_evals[0] + prev_evals[1]);
            debug!("rest =\n{:?}", claim.get_result());
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_prover_messages[0])
            .unwrap();

        // --- For round 1 < i < n, perform the check ---
        // g_{i - 1}(r_i) = g_i(0) + g_i(1)
        for curr_evals in sumcheck_prover_messages.iter().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();

            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            transcript
                .append_field_elements("Sumcheck evaluations", curr_evals)
                .unwrap();

            prev_evals = curr_evals;
            challenges.push(challenge);
        }

        // --- In the final round, we check that g(r_1, ..., r_n) = g_n(r_n) ---
        // Here, we first sample r_n.
        let final_chal = transcript
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        // --- This automatically asserts that the expression is fully bound and simply ---
        // --- attempts to combine/collect the expression evaluated at the (already bound) challenge coords ---
        let expr_evaluated_at_challenge_coord = gather_combine_all_evals(&self.expression).unwrap();

        // --- Simply computes \beta((g_1, ..., g_n), (u_1, ..., u_n)) for claim coords (g_1, ..., g_n) and ---
        // --- bound challenges (u_1, ..., u_n) ---
        let beta_fn_evaluated_at_challenge_point =
            compute_beta_over_two_challenges(claim.get_point(), &challenges);

        // --- The actual value should just be the product of the two ---
        let mle_evaluated_at_challenge_coord =
            expr_evaluated_at_challenge_coord * beta_fn_evaluated_at_challenge_point;

        // --- Computing g_n(r_n) ---
        let g_n_evaluated_at_r_n =
            evaluate_at_a_point(prev_evals, final_chal).map_err(LayerError::InterpError)?;

        // --- Checking the two against one another ---
        if mle_evaluated_at_challenge_coord != g_n_evaluated_at_r_n {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    fn get_claims(&self) -> Result<Vec<Claim<F>>, LayerError> {

        // First off, parse the expression that is associated with the layer...
        // Next, get to the actual claims that are generated by each expression and grab them
        // Return basically a list of (usize, Claim)
        let layerwise_expr = &self.expression;

        // --- Define how to parse the expression tree ---
        // - Basically we just want to go down it and pass up claims
        // - We can only add a new claim if we see an MLE with all its indices bound

        let mut claims: Vec<Claim<F>> = Vec::new();

       

        let mut observer_fn = |exp: &ExpressionStandard<F>| {
            match exp {
                ExpressionStandard::Mle(mle_ref) => {
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
                        fixed_mle_indices.push(mle_idx.val().ok_or(ClaimError::MleRefMleError)?);
                    }

                    // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                    let mle_layer_id = mle_ref.get_layer_id();

                    // --- Grab the actual value that the claim is supposed to evaluate to ---
                    if mle_ref.bookkeeping_table().len() != 1 {
                        dbg!(&mle_ref.bookkeeping_table);
                        return Err(ClaimError::MleRefMleError);
                    }
                    let claimed_value = mle_ref.bookkeeping_table()[0];

                    // --- Construct the claim ---
                    // println!("========\n I'm making a GKR layer claim for an MLE!!\n==========");
                    // println!("From: {:#?}, To: {:#?}", self.id().clone(), mle_layer_id);
                    let claim: Claim<F> = Claim::new(
                        fixed_mle_indices,
                        claimed_value,
                        Some(self.id().clone()),
                        Some(mle_layer_id),
                        Some(MleEnum::Dense(mle_ref.clone())),
                    );

                    // --- Push it into the list of claims ---
                    claims.push(claim);
                }
                ExpressionStandard::Product(mle_refs) => {
                    for mle_ref in mle_refs {
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
                            return Err(ClaimError::MleRefMleError);
                        }
                        let claimed_value = mle_ref.bookkeeping_table()[0];

                        // --- Construct the claim ---
                        // need to populate the claim with the mle ref we are grabbing the claim from
                        let claim: Claim<F> = Claim::new(
                            fixed_mle_indices,
                            claimed_value,
                            Some(self.id().clone()),
                            Some(mle_layer_id),
                            Some(MleEnum::Dense(mle_ref.clone()))
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

    fn id(&self) -> &LayerId {
        &self.id
    }

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
        let num_vars = expr.index_mle_indices(0);
        let degree = get_round_degree(&expr, 0);
        // expr.init_beta_tables(prev_layer_claim);
        let mut num_evals = (num_vars) * (num_claims); //* degree;

        if ENABLE_OPTIMIZATION {
            let mut degree_reduction = num_vars as i64;
            for j in 0..num_vars {
                for i in 1..num_claims {
                    if claim_vecs[i][j] != claim_vecs[i - 1][j] {
                        degree_reduction -= 1;
                        break;
                    }
                }
            }
            assert!(degree_reduction >= 0);

            // Evaluate the P(x) := W(l(x)) polynomial at deg(P) + 1
            // points. W : F^n -> F is a multi-linear polynomial on
            // `num_vars` variables and l : F -> F^n is a canonical
            // polynomial passing through `num_claims` points so its degree is
            // at most `num_claims - 1`. This imposes an upper
            // bound of `num_vars * (num_claims - 1)` to the degree of P.
            // However, the actual degree of P might be lower.
            // For any coordinate `i` such that all claims agree
            // on that coordinate, we can quickly deduce that `l_i(x)` is a
            // constant polynomial of degree zero instead of `num_claims -
            // 1` which brings down the total degree by the same amount.
            num_evals =
                (num_vars) * (num_claims - 1) + 1 - (degree_reduction as usize) * (num_claims - 1);
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

                let wlx_eval_on_mle_ref = combine_mle_refs_with_aggregate(&claim_mle_refs, &new_chal);
                wlx_eval_on_mle_ref.unwrap()
            })
            .collect();

        // concat this with the first k evaluations from the claims to
        // get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        Ok(wlx_evals)
    }

    fn get_enum(self) -> LayerEnum<F, Self::Transcript> {
        LayerEnum::Gkr(self)
    }
}

/// The builder type for a Layer
pub trait LayerBuilder<F: FieldExt> {
    /// The layer that makes claims on this layer in the GKR protocol. The next layer in the GKR protocol
    type Successor;

    /// Build the expression that will be sumchecked
    fn build_expression(&self) -> ExpressionStandard<F>;

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
    EFn: Fn(&M) -> ExpressionStandard<F>,
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

    fn build_expression(&self) -> ExpressionStandard<F> {
        let first = self.first.build_expression();
        let second = self.second.build_expression();

        // return first.concat_expr(second);

        let zero_expression: ExpressionStandard<F> = ExpressionStandard::Constant(F::zero());

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
        // ExpressionStandard::Selector(MleIndex::Iterated, Box::new(first_padded), Box::new(second_padded))
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
        EFn: Fn(&M) -> ExpressionStandard<F>,
        S,
        LFn: Fn(&M, LayerId, Option<Vec<MleIndex<F>>>) -> S,
    > LayerBuilder<F> for SimpleLayer<M, EFn, LFn>
{
    type Successor = S;

    fn build_expression(&self) -> ExpressionStandard<F> {
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
    //             ExpressionStandard::Mle(mle1.mle_ref()) + ExpressionStandard::Mle(mle2.mle_ref())
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

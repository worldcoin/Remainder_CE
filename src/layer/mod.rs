//! A layer is a combination of multiple MLEs with an expression

pub mod claims;
// mod gkr_layer;

use std::{marker::PhantomData, iter};

use thiserror::Error;

use crate::{
    expression::{Expression, ExpressionError, ExpressionStandard},
    mle::{
        beta::{BetaError, BetaTable, evaluate_beta},
        dense::{DenseMle, DenseMleRef},
        MleIndex, MleRef,
    },
    sumcheck::{compute_sumcheck_message, get_round_degree, SumOrEvals, evaluate_at_a_point, InterpError},
    FieldExt,
    transcript::Transcript,
    prover::SumcheckProof,
};

use self::claims::ClaimError;

pub type Claim<F> = (Vec<F>, F);

#[derive(Error, Debug, Clone)]
pub enum LayerError {
    #[error("Layer isn't ready to prove")]
    LayerNotReady,
    #[error("Error with underlying expression: {0}")]
    ExpressionError(ExpressionError),
    #[error("Error with aggregating curr layer")]
    AggregationError,
    #[error("Error with getting Claim: {0}")]
    ClaimError(ClaimError),
    #[error("Error with verifying layer: {0}")]
    VerificationError(VerificationError),
    #[error("Beta Error: {0}")]
    BetaError(BetaError),
    #[error("InterpError: {0}")]
    InterpError(InterpError)
}

#[derive(Error, Debug, Clone)]
pub enum VerificationError {
    #[error("The sum of the first evaluations do not equal the claim")]
    SumcheckStartFailed,
    #[error("The sum of the current rounds evaluations do not equal the previous round at a random point")]
    SumcheckFailed,
    #[error("The final rounds evaluations at r do not equal the oracle query")]
    FinalSumcheckFailed,
    #[error("The Oracle query does not match the final claim")]
    GKRClaimCheckFailed
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
///The location of a layer within the GKR circuit
pub enum LayerId {
    ///An Mle located in the input layer
    Input,
    ///A layer within the GKR protocol, indexed by it's layer id
    Layer(usize),
}

///A layer is what you perform sumcheck over, it is made up of an expression and MLEs that contribute evaluations to that expression
pub trait Layer<F: FieldExt> {
    type Transcript: Transcript<F>;
    ///The Expression type that this Layer is defined by
    // type Expression: Expression<F>;

    ///Injest a claim, initialize beta tables, and do any other bookeeping that needs to be done before the sumcheck starts
    fn start_sumcheck(&mut self, claim: Claim<F>) -> Result<(Vec<F>, usize), LayerError> {
        let (max_round, beta) = {
            let (expression, _) = self.mut_expression_and_beta();

            let mut beta = BetaTable::new(claim).map_err(|err| LayerError::BetaError(err))?;

            let max_round = std::cmp::max(expression.index_mle_indices(0), beta.table.index_mle_indices(0));
            (max_round, beta)
        };

        self.set_beta(beta);

        let (expression, beta) = self.mut_expression_and_beta();

        let beta = beta.as_ref().unwrap();

        let degree = get_round_degree(expression, 0);

        let eval = compute_sumcheck_message(expression, 0, degree, &beta)
            .map_err(LayerError::ExpressionError)?;

        let out = if let SumOrEvals::Evals(evals) = eval {
                Ok(evals)
            } else {
                Err(LayerError::ExpressionError(
                    ExpressionError::EvaluationError(
                        "Received a sum variant from evaluate expression before the final round",
                    ),
                ))
            }?;

        Ok((out, max_round))
    }

    ///Computes a round of the sumcheck protocol on this Layer
    fn prove_round(&mut self, round_index: usize, challenge: F) -> Result<Vec<F>, LayerError> {
        let (expression, beta) = self.mut_expression_and_beta();
        let beta = beta.as_mut().ok_or(LayerError::LayerNotReady)?;
        expression.fix_variable(round_index - 1, challenge);
        beta.beta_update(round_index - 1, challenge).map_err(LayerError::BetaError)?;

        // --- Grabs the degree of univariate polynomial we are sending over ---
        let degree = get_round_degree(expression, round_index);

        let eval = compute_sumcheck_message(expression, round_index, degree, beta)
            .map_err(LayerError::ExpressionError)?;

        if let SumOrEvals::Evals(evals) = eval {
            Ok(evals)
        } else {
            Err(LayerError::ExpressionError(
                ExpressionError::EvaluationError(
                    "Received a sum variant from evaluate expression before the final round",
                ),
            ))
        }
    }

    /// Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self, 
        claim: Claim<F>, 
        sumcheck_rounds: Vec<Vec<F>>, 
        transcript: &mut Self::Transcript, 
    ) -> Result<(), LayerError>{

        let mut challenges = vec![];
        let (expression, _) = self.mut_expression_and_beta();
        
        // first round, see Thaler book page 34
        let mut prev_evals = &sumcheck_rounds[0];
        let claimed_claim = prev_evals[0] + prev_evals[1];
        dbg!(claimed_claim, claim.1);
        if prev_evals[0] + prev_evals[1] != claim.1 {
            return Err(LayerError::VerificationError(VerificationError::SumcheckStartFailed));
        }

        transcript
            .append_field_elements("Initial Sumcheck evaluations", &sumcheck_rounds[0])
            .unwrap();

        // round j, 1 < j < v
        for curr_evals in sumcheck_rounds.iter().skip(1) {
            let challenge = transcript.get_challenge("Sumcheck challenge").unwrap();

            let prev_at_r = evaluate_at_a_point(prev_evals, challenge).map_err(|err| LayerError::InterpError(err))?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(VerificationError::SumcheckFailed));
            };

            transcript
            .append_field_elements("Sumcheck evaluations", &curr_evals)
            .unwrap();

            prev_evals = curr_evals;
            challenges.push(challenge);
        }
        // final round v
        let final_chal = transcript.get_challenge("Final Sumcheck challenge").unwrap();
        challenges.push(final_chal);

        let claimed_value = claim.1;

        // uses the expression to make one single oracle query
        let mut beta = BetaTable::new(claim).unwrap();
        let _ = expression.index_mle_indices(0);
        let mle_bound = expression.evaluate_expr(challenges.clone()).unwrap();
        let beta_bound = evaluate_beta(&mut beta, challenges).unwrap();
        let oracle_query = mle_bound * beta_bound;

        // if oracle_query != claimed_value {
        //     return Err(LayerError::VerificationError(VerificationError::GKRClaimCheckFailed));
        // }

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).map_err(|err| LayerError::InterpError(err))?;
        if oracle_query != prev_at_r {
            return Err(LayerError::VerificationError(VerificationError::FinalSumcheckFailed));
        }

        // transcript
        // .append_field_elements("Sumcheck evaluations", &prev_evals)
        // .unwrap();

        Ok(())
    }

    ///Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<(LayerId, Claim<F>)>, LayerError> {
        // First off, parse the expression that is associated with the layer...
        // Next, get to the actual claims that are generated by each expression and grab them
        // Return basically a list of (usize, Claim)
        let layerwise_expr = self.get_expression();

        // --- Define how to parse the expression tree ---
        // - Basically we just want to go down it and pass up claims
        // - We can only add a new claim if we see an MLE with all its indices bound

        let mut claims: Vec<Claim<F>> = Vec::new();
        let mut indices: Vec<LayerId> = Vec::new();

        let mut observer_fn = |exp: &ExpressionStandard<F>| {
            match exp {
                ExpressionStandard::Mle(mle_ref) => {
                    // --- First ensure that all the indices are fixed ---
                    let mle_indices = mle_ref.mle_indices();

                    // --- This is super jank ---
                    let mut fixed_mle_indices: Vec<F> = vec![];
                    for mle_idx in mle_indices {
                        match mle_idx {
                            // --- We can't have either iterated or indexed bits ---
                            MleIndex::IndexedBit(_) | MleIndex::Iterated => {
                                return Err(ClaimError::MleRefMleError);
                            }
                            // --- We can't have either iterated or indexed bits ---
                            MleIndex::Bound(idx) => {
                                fixed_mle_indices.push(*idx);
                            }
                            MleIndex::Fixed(one) => {
                                let idx = if *one { F::one() } else { F::zero() };
                                fixed_mle_indices.push(idx);
                            }
                        }
                    }

                    // --- Grab the layer ID (i.e. MLE index) which this mle_ref refers to ---
                    let mle_layer_id = match mle_ref.get_layer_id() {
                        None => {
                            return Err(ClaimError::LayerMleError);
                        }
                        Some(layer_id) => layer_id,
                    };

                    // --- Grab the actual value that the claim is supposed to evaluate to ---
                    if mle_ref.bookkeeping_table().len() != 1 {
                        return Err(ClaimError::MleRefMleError);
                    }
                    // TODO(ryancao): Does this accidentally take ownership of that element?
                    // Answer: No, because F implements Copy
                    let claimed_value = mle_ref.bookkeeping_table()[0];

                    // --- Construct the claim ---
                    let claim: Claim<F> = (fixed_mle_indices, claimed_value);

                    // --- Push it into the list of claims ---
                    // --- Also push the layer_id ---
                    claims.push(claim);
                    indices.push(mle_layer_id);
                }
                _ => {}
            }
            Ok(())
        };

        // TODO!(ryancao): What the heck is this code doing?
        layerwise_expr.traverse(&mut observer_fn).map_err(|err| LayerError::ClaimError(err))?;

        Ok(indices.into_iter().zip(claims).collect())
    }

    ///Gets the unique id of this layer
    fn get_id(&self) -> &LayerId;

    ///Get the master expression associated with this Layer
    fn get_expression(&self) -> &ExpressionStandard<F>;

    ///Get the master expression associated with this Layer mutably
    fn mut_expression_and_beta(&mut self) -> (&mut ExpressionStandard<F>, &mut Option<BetaTable<F>>);

    ///Initializes the beta table
    fn set_beta(&mut self, beta: BetaTable<F>);

    ///Get beta table
    fn beta(&self) -> &Option<BetaTable<F>>;

    ///Create new ConcreteLayer from a LayerBuilder
    fn new<L: LayerBuilder<F>>(builder: L, id: LayerId) -> Self
    where
        Self: Sized;
}

///Default Layer abstraction
pub struct GKRLayer<F: FieldExt, Tr: Transcript<F>> {
    id: LayerId,
    expression: ExpressionStandard<F>,
    beta: Option<BetaTable<F>>,
    _marker: PhantomData<Tr>,
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

    fn get_expression(&self) -> &ExpressionStandard<F> {
        &self.expression
    }

    fn mut_expression_and_beta(&mut self) -> (&mut ExpressionStandard<F>, &mut Option<BetaTable<F>>) {
        (&mut self.expression, &mut self.beta)
    }

    fn get_id(&self) -> &LayerId {
        &self.id
    }

    fn set_beta(&mut self, beta: BetaTable<F>) {
        self.beta = Some(beta);
    }

    fn beta(&self) -> &Option<BetaTable<F>> {
        &self.beta
    }
}

///The builder type for a Layer
pub trait LayerBuilder<F: FieldExt> {
    ///The layer that makes claims on this layer in the GKR protocol. The next layer in the GKR protocol
    type Successor;

    ///Build the expression that will be sumchecked
    fn build_expression(&self) -> ExpressionStandard<F>;

    ///Generate the next layer 
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor ;

    ///Concatonate two layers together
    fn concat<Other: LayerBuilder<F>>(self, rhs: Other) -> ConcatLayer<F, Self, Other>
    where
        Self: Sized,
    {
        ConcatLayer {
            first: self,
            second: rhs,
            _marker: PhantomData,
        }
    }
}

    ///Creates a simple layer from an mle, with closures for defining how the mle turns into an expression and a next layer
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


///The layerbuilder that represents two layers concatonated together
pub struct ConcatLayer<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> {
    first: A,
    second: B,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, A: LayerBuilder<F>, B: LayerBuilder<F>> LayerBuilder<F> for ConcatLayer<F, A, B> {
    type Successor = (A::Successor, B::Successor);

    fn build_expression(&self) -> ExpressionStandard<F> {
        let first = self.first.build_expression();
        let second = self.second.build_expression();
        first.concat(second)
    }

    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        (
            self.first.next_layer(
                id.clone(),
                Some(
                    prefix_bits
                        .clone()
                        .into_iter()
                        .flatten()
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
                        .chain(std::iter::once(MleIndex::Fixed(false)))
                        .collect(),
                ),
            ),
        )
    }
}

///A simple layer defined ad-hoc with two closures
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
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand::rngs::OsRng;

    use crate::{mle::{dense::DenseMle, MleIndex}, expression::ExpressionStandard, sumcheck::{dummy_sumcheck, verify_sumcheck_messages}, transcript::poseidon_transcript::PoseidonTranscript};

    use super::{from_mle, GKRLayer, Layer, LayerId, LayerBuilder};

    #[test]
    fn build_simple_layer() {
        let mut rng = test_rng();
        let mle1 = DenseMle::<Fr, Fr>::new(vec![Fr::from(2), Fr::from(3), Fr::from(6), Fr::from(7)]);
        let mle2 = DenseMle::<Fr, Fr>::new(vec![Fr::from(3), Fr::from(1), Fr::from(9), Fr::from(2)]);

        let builder = from_mle((mle1, mle2), |(mle1, mle2)| {
            ExpressionStandard::Mle(mle1.mle_ref()) + ExpressionStandard::Mle(mle2.mle_ref())
        }, |(mle1, mle2), _, _: Option<Vec<MleIndex<Fr>>>| {
            mle1.clone().into_iter().zip(mle2.clone().into_iter()).map(|(first, second)| first + second).collect::<DenseMle<_, _>>()
        });

        let next: DenseMle<Fr, Fr> = builder.next_layer(LayerId::Layer(0), None);

        let mut layer = GKRLayer::<_, PoseidonTranscript<Fr>>::new(builder, LayerId::Layer(0));

        let sum = dummy_sumcheck(&mut layer.expression, &mut rng, todo!());
        verify_sumcheck_messages(sum, layer.expression, todo!(), &mut OsRng).unwrap();

        
    }
}
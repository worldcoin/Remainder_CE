use crate::{
    hyrax_primitives::{
        proof_of_claim_agg::ProofOfClaimAggregation, proof_of_product::ProofOfProduct,
        proof_of_sumcheck::ProofOfSumcheck,
    },
    pedersen::{CommittedScalar, CommittedVector, PedersenCommitter},
    utils::vandermonde::VandermondeInverse,
};
use ark_std::iterable::Iterable;
use itertools::Itertools;
use rand::Rng;
use remainder::claims::wlx_eval::YieldWLXEvals;
use remainder::layer::combine_mle_refs::get_og_mle_refs;
use remainder::layer::product::{new_with_values, Product};
use remainder::layer::product::{Intermediate, PostSumcheckLayer};
use remainder::layer::{LayerId, PostSumcheckEvaluation, SumcheckLayer};
use remainder::mle::mle_enum::MleEnum;
use remainder::{claims::wlx_eval::claim_group::ClaimGroup, layer::layer_enum::CircuitLayerEnum};
use remainder::{claims::wlx_eval::ClaimMle, layer::CircuitLayer};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::halo2curves::group::ff::Field;
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};
use remainder_shared_types::FieldExt;
/// This struct represents what a proof looks like for one layer of GKR, but Hyrax version.
pub struct HyraxLayerProof<C: PrimeOrderCurve> {
    /// This is the proof of the sumcheck rounds for that layer.
    pub proof_of_sumcheck: ProofOfSumcheck<C>,
    /// The values of the claims made by this layer, in commitment form.
    pub commitments: Vec<C>,
    /// This is the associated proof of products for the final commitments of that layer.
    pub proofs_of_product: Vec<ProofOfProduct<C>>,
    /// This is the proof of claim aggregation for the associated claims that were
    /// aggregated to make a claim on this layer.
    pub proof_of_claim_agg: ProofOfClaimAggregation<C>,
}

impl<C: PrimeOrderCurve> HyraxLayerProof<C> {
    /// Generate the associated \alpha_i commitment for round i. This is
    /// essentially the coefficients to the univariate committed to (using the appropriate
    /// generators for that round, i.e. accounting for the zero padding of the sumcheck messages).
    fn commit_to_round(
        underlying_layer: &mut (impl SumcheckLayer<C::Scalar>
                  + PostSumcheckEvaluation<C::Scalar>
                  + YieldWLXEvals<C::Scalar>),
        committer: &PedersenCommitter<C>,
        round_index: usize,
        max_degree: usize,
        num_rounds: usize,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> CommittedVector<C> {
        // The univariate evaluations for that round (i.e. f(0), f(1), ...)
        let round_evaluations = underlying_layer
            .compute_round_sumcheck_message(round_index)
            .unwrap();

        // Convert the evaluations above into coefficients (in Hyrax, it is the coefficients, not
        // the evaluations, that are used)
        let mut round_coefficients = vec![C::Scalar::ZERO; (max_degree + 1) * round_index];
        // Pad the coefficients to form the $v^{(`round_index`)}$ vector, so that only the generators for this round are used
        let computed_coeffs = converter.convert_to_coefficients(round_evaluations);
        round_coefficients.extend(&computed_coeffs);
        round_coefficients.extend(vec![
            C::Scalar::ZERO;
            (max_degree + 1) - computed_coeffs.len()
        ]);
        round_coefficients.extend(vec![
            C::Scalar::ZERO;
            (max_degree + 1) * (num_rounds - round_index - 1)
        ]);
        let blinding_factor = C::Scalar::random(blinding_rng);
        let commitment = committer.committed_vector(&round_coefficients, &blinding_factor);
        commitment
    }

    /// Produce a [HyraxLayerProof] for a given layer, given the unaggregated claims on that layer.
    /// Return also a [HyraxClaim] representing the aggregated claim.
    pub fn prove(
        // The layer that we are proving
        mut layer: &mut (impl SumcheckLayer<C::Scalar>
                  + PostSumcheckEvaluation<C::Scalar>
                  + YieldWLXEvals<C::Scalar>),
        // The claims on that layer (unaggregated)
        claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        mut blinding_rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> (Self, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>) {
        let interpolant_coeffs = if claims.len() > 1 {
            // NB we don't use aggregate_claims here because the sampling of the evaluation
            // point for the aggregate claim needs to happen elsewhere in Hyrax.
            // Convert to a ClaimGroup so that we can use the helper functions
            let claim_group = ClaimGroup::new(
                claims
                    .iter()
                    .map(|hyrax_claim| hyrax_claim.to_claim())
                    .collect_vec(),
            )
            .unwrap();
            // Calculate the evaluations at 0, 1, 2, ..
            let wlx_evals = layer
                .get_wlx_evaluations(
                    claim_group.get_claim_points_matrix(),
                    claim_group.get_results(),
                    get_og_mle_refs(claim_group.get_claim_mle_refs()),
                    claim_group.get_num_claims(),
                    claim_group.get_num_vars(),
                )
                .unwrap();
            // Convert the evaluations to coefficients
            converter.convert_to_coefficients(wlx_evals)
        } else {
            vec![claims[0].evaluation.value]
        };

        let (proof_of_claim_agg, agg_claim) = ProofOfClaimAggregation::prove(
            &claims,
            &interpolant_coeffs,
            &committer,
            &mut blinding_rng,
            transcript,
        );
        // Note that the commitment to the aggregate evaluationp `eval` does not need to be added to the
        // transcript since it is derived from commitments that are added to the transcript already
        // (the commitments to the coefficients of the interpolant).
        let degree = layer.max_degree();
        // The number of sumcheck rounds for this layer
        let num_rounds = layer.num_sumcheck_rounds();

        // These are going to be the commitments to the sumcheck messages.
        let mut messages: Vec<CommittedVector<C>> = vec![];
        // These are the challenges for this layer of sumcheck. Append as we go.
        let mut bindings: Vec<C::Scalar> = vec![];

        // Initialize the sumcheck layer.
        layer.initialize_sumcheck(&agg_claim.point);

        // Go through each of the sumcheck rounds and produce the \alpha_i messages.
        (0..num_rounds).for_each(|round| {
            let round_commit = HyraxLayerProof::commit_to_round(
                layer,
                &committer,
                round,
                degree,
                num_rounds,
                &mut blinding_rng,
                converter,
            );
            messages.push(round_commit);
            transcript.append_ec_point("sumcheck message commitment", round_commit.commitment);

            let challenge = transcript.get_scalar_field_challenge("sumcheck round challenge");
            bindings.push(challenge);
            layer.bind_round_variable(round, challenge);
        });

        // Get the post sumcheck layer
        let post_sumcheck_layer = layer.get_post_sumcheck_layer(&bindings, &agg_claim.point);

        // Commit to all the necessary values
        let post_sumcheck_layer_committed =
            commit_to_post_sumcheck_layer(&post_sumcheck_layer, &committer, &mut blinding_rng);

        // Get the commitments (i.e. points on C)
        let commitments =
            committed_scalar_psl_as_commitments(post_sumcheck_layer_committed).get_values();

        // Add each of the commitments to the transcript
        transcript.append_ec_points("commitment to product input/outputs", &commitments);

        // Get the claims made in this layer
        let committed_claims: Vec<_> = post_sumcheck_layer_committed
            .0
            .iter()
            .map(|product| get_claims_from_product(&product))
            .flatten()
            .collect();

        // Proof of sumcheck
        // Note that product_evaluations have already been added to the transcript (along with the rest of the commitments)
        let proof_of_sumcheck = ProofOfSumcheck::prove(
            &agg_claim.evaluation,
            &messages,
            degree,
            &post_sumcheck_layer_committed,
            &bindings,
            &committer,
            blinding_rng,
            transcript,
        );

        // perform the proof of products
        let proofs_of_products = post_sumcheck_layer_committed
            .0
            .iter()
            .filter_map(|product| product.get_product_triples())
            .flatten()
            .map(|(x, y, z)| {
                ProofOfProduct::prove(&x, &y, &z, &committer, &mut blinding_rng, transcript)
            })
            .collect();

        (
            HyraxLayerProof {
                proof_of_sumcheck,
                commitments,
                proofs_of_product: proofs_of_products,
                proof_of_claim_agg,
            },
            // Note that we don't need to add the claims to the transcript, since they were added
            // above along with all other commitments
            committed_claims,
        )
    }

    /// Verify the [HyraxLayerProof] given commitments to the unaggregated claims, returning a
    /// commitment to the aggregated claim in the form of a [HyraxClaim]. and Vec<[ProofOfProduct]>
    /// given a [HyraxLayerProof]. Takes a copy of the layer as well, though this is just to know
    /// its structure (not the particular MLE values).
    pub fn verify(
        proof: &HyraxLayerProof<C>,
        // a description of the layer being proven
        layer_desc: &CircuitLayerEnum<C::Scalar>,
        // commitments to the unaggregated claims
        claim_commitments: &Vec<HyraxClaim<C::Scalar, C>>,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) -> Vec<HyraxClaim<C::Scalar, C>> {
        let HyraxLayerProof {
            proof_of_claim_agg,
            commitments,
            proofs_of_product,
            proof_of_sumcheck,
        } = proof;

        // Verify the proof of claim aggregation
        let agg_claim = proof_of_claim_agg.verify(&claim_commitments, &committer, transcript);

        // The number of sumcheck rounds w.r.t. to the beta table rather than just the expression.
        // Because the beta table number of variables is exactly the number of points in the claim
        // made on that layer, we take the max of the number of variables in the expression and
        // the number of variables in the beta table.
        let num_sumcheck_rounds_expected = layer_desc.num_sumcheck_rounds();

        // Verify the proof of sumcheck
        // Add first sumcheck message to transcript, which is the proported sum.
        if num_sumcheck_rounds_expected > 0 {
            let transcript_first_sumcheck_message = transcript
                .consume_ec_point("first sumcheck message commitment")
                .unwrap();

            assert_eq!(
                transcript_first_sumcheck_message,
                proof_of_sumcheck.messages[0]
            );
        }

        // Collect the "bindings" for each of the sumcheck rounds. Add sumcheck messages to transcript.
        let mut bindings: Vec<C::Scalar> = vec![];
        proof_of_sumcheck
            .messages
            .iter()
            .skip(1)
            .for_each(|message| {
                let challenge = transcript
                    .get_scalar_field_challenge("sumcheck round challenge")
                    .unwrap();
                bindings.push(challenge);

                let transcript_sumcheck_message_commit = transcript
                    .consume_ec_point("sumcheck message commitment")
                    .unwrap();
                assert_eq!(&transcript_sumcheck_message_commit, message);
            });

        // Final challenge in sumcheck -- needed for "oracle query".
        if num_sumcheck_rounds_expected > 0 {
            let final_chal = transcript
                .get_scalar_field_challenge("sumcheck round challenge")
                .unwrap();
            bindings.push(final_chal);
        }

        // Verify that we have the correct number of bindings
        assert_eq!(bindings.len(), num_sumcheck_rounds_expected);

        // Add the commitments made by the prover to the transcript
        let transcript_commitments: Vec<C> = transcript
            .consume_ec_points("commitment to product input/outputs", commitments.len())
            .unwrap();
        assert_eq!(&transcript_commitments, commitments);

        // Build the [PostSumcheckLayer] from the commitments and the layer description
        let verifier_expr = layer_desc
            .expression
            .bind(&point, transcript_reader)
            .map_err(|err| VerificationError::ExpressionError(err))?;

        let verifier_layer = VerifierRegularLayer::new_raw(self.layer_id(), verifier_expr);

        let post_sumcheck_layer_desc = layer_desc
            .layer_desc_enum
            .get_post_sumcheck_layer(&bindings, &agg_claim.point);
        let post_sumcheck_layer: PostSumcheckLayer<C::Scalar, C> =
            new_with_values(&post_sumcheck_layer_desc, commitments);

        // Verify the proof of sumcheck!
        proof_of_sumcheck.verify(
            &agg_claim.evaluation,
            layer_desc.max_degree,
            &post_sumcheck_layer,
            &bindings,
            &committer,
            transcript,
        );

        // Extract the triples of commitments that must be proven in the proof of product
        // and verify the proofs of product
        let product_triples: Vec<(C, C, C)> = post_sumcheck_layer
            .0
            .iter()
            .filter_map(|commitment| commitment.get_product_triples())
            .flatten()
            .collect_vec();
        assert_eq!(product_triples.len(), proofs_of_product.len());
        product_triples
            .iter()
            .zip(proofs_of_product.iter())
            .for_each(|((x, y, z), proof)| {
                proof.verify(&x, &y, &z, &committer, transcript);
            });

        // Extract the claims that the prover implicitly made on other layers by sending `commitments`.
        let claims = post_sumcheck_layer
            .0
            .iter()
            .map(|product| get_claims_from_product(&product))
            .flatten()
            .collect_vec();

        claims
    }
}

// ---------- This is where all the Hyrax [PostSumcheckLayer]-specific stuff is going! ----------
/// Evaluate the PostSumcheckLayer to a single CommittedScalar.
pub fn evaluate_committed_scalar<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, CommittedScalar<C>>,
) -> CommittedScalar<C> {
    post_sumcheck_layer
        .0
        .iter()
        .fold(CommittedScalar::zero(), |acc, product| {
            acc + product.get_result() * product.coefficient
        })
}

/// Turn all the CommittedScalars into commitments i.e. Cs.
pub fn committed_scalar_psl_as_commitments<C: PrimeOrderCurve>(
    post_sumcheck_layer: PostSumcheckLayer<C::Scalar, CommittedScalar<C>>,
) -> PostSumcheckLayer<C::Scalar, C> {
    PostSumcheckLayer(
        post_sumcheck_layer
            .0
            .iter()
            .map(|product| {
                let commitments = product
                    .intermediates
                    .iter()
                    .map(|pp| match pp {
                        Intermediate::Atom {
                            layer_id,
                            point,
                            value,
                            mle_enum: _,
                        } => Intermediate::Atom {
                            layer_id: *layer_id,
                            point: point.clone(),
                            value: value.commitment.clone(),
                            mle_enum: None,
                        },
                        Intermediate::Composite { value } => Intermediate::Composite {
                            value: value.commitment.clone(),
                        },
                    })
                    .collect();
                Product {
                    coefficient: product.coefficient,
                    intermediates: commitments,
                }
            })
            .collect(),
    )
}

/// Evaluate the PostSumcheckLayer to a single scalar
pub fn evaluate_committed_psl<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, C>,
) -> C {
    post_sumcheck_layer
        .0
        .iter()
        .fold(C::zero(), |acc, product| {
            acc + product.get_result() * product.coefficient
        })
}

/// Return the claims made on other layers by the atomic factors of this product.
pub fn get_claims_from_product<F: FieldExt, T: Clone>(
    product: &Product<F, T>,
) -> Vec<HyraxClaim<F, T>> {
    product
        .intermediates
        .iter()
        .filter_map(|pp| match pp {
            Intermediate::Atom {
                layer_id,
                point,
                value,
                mle_enum,
            } => Some(HyraxClaim {
                to_layer_id: *layer_id,
                mle_enum: mle_enum.clone(),
                point: point.clone(),
                evaluation: value.clone(),
            }),
            Intermediate::Composite { .. } => None,
        })
        .collect()
}

/// Implementation of HyraxClaim as used by the prover
impl<C: PrimeOrderCurve> HyraxClaim<C::Scalar, CommittedScalar<C>> {
    /// Convert to a raw [Claim] for claim aggregation
    pub fn to_claim(&self) -> ClaimMle<C::Scalar> {
        let mut claim = ClaimMle::new_raw(self.point.clone(), self.evaluation.value);
        claim.to_layer_id = Some(self.to_layer_id.clone());
        claim.mle_ref = self.mle_enum.clone();
        claim
    }

    /// Convert to a HyraxClaim<C::Scalar, C>
    pub fn to_claim_commitment(&self) -> HyraxClaim<C::Scalar, C> {
        HyraxClaim {
            point: self.point.clone(),
            to_layer_id: self.to_layer_id,
            mle_enum: self.mle_enum.clone(),
            evaluation: self.evaluation.commitment,
        }
    }
}

/// Represents a claim made on a layer by an atomic factor of a product.
/// T could be:
///     C::Scalar (if used by the prover), or
///     CommittedScalar<C> (if used by the prover)
///     to interface with claim aggregation code in remainder
///     C (this is the verifier's view, i.e. just the commitment)
#[derive(Clone, Debug)]
pub struct HyraxClaim<F: FieldExt, T> {
    /// Id of the layer upon which the claim is made
    pub to_layer_id: LayerId,
    /// The evaluation point
    pub point: Vec<F>,
    /// The original mle_enum (or None)
    pub mle_enum: Option<MleEnum<F>>,
    /// The value of the claim
    pub evaluation: T,
}

/// Returns a CommittedScalar version of the PostSumcheckLayer.
pub fn commit_to_post_sumcheck_layer<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut blinding_rng: &mut impl Rng,
) -> PostSumcheckLayer<C::Scalar, CommittedScalar<C>> {
    PostSumcheckLayer(
        post_sumcheck_layer
            .0
            .iter()
            .map(|product| commit_to_product(&product, committer, &mut blinding_rng))
            .collect(),
    )
}

// Helper for commit_to_post_sumcheck_layer.
// Returns a CommittedScalar version of the Product.
fn commit_to_product<C: PrimeOrderCurve>(
    product: &Product<C::Scalar, C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut blinding_rng: &mut impl Rng,
) -> Product<C::Scalar, CommittedScalar<C>> {
    let committed_scalars = product
        .intermediates
        .iter()
        .map(|pp| match pp {
            Intermediate::Atom {
                layer_id,
                point,
                value,
                mle_enum,
            } => Intermediate::Atom {
                layer_id: *layer_id,
                point: point.clone(),
                value: committer.committed_scalar(value, &C::Scalar::random(&mut blinding_rng)),
                mle_enum: mle_enum.clone(),
            },
            Intermediate::Composite { value } => Intermediate::Composite {
                value: committer.committed_scalar(value, &C::Scalar::random(&mut blinding_rng)),
            },
        })
        .collect();
    Product {
        intermediates: committed_scalars,
        coefficient: product.coefficient,
    }
}

use crate::{
    hyrax_primitives::{
        proof_of_claim_agg::ProofOfClaimAggregation, proof_of_product::ProofOfProduct,
        proof_of_sumcheck::ProofOfSumcheck,
    },
    utils::vandermonde::VandermondeInverse,
};
use itertools::Itertools;
use rand::{CryptoRng, Rng, RngCore};
use remainder::layer::product::{new_with_values, PostSumcheckLayerTree};
use remainder::layer::Layer;
use remainder::layer::LayerDescription;
use remainder::layer::LayerId;
use remainder::mle::dense::DenseMle;
use remainder::{claims::claim_aggregation::get_wlx_evaluations, layer::layer_enum::LayerEnum};
use remainder::{
    claims::claim_group::ClaimGroup, layer::combine_mles::get_indexed_layer_mles_to_combine,
};
use remainder::claims::RawClaim;
use remainder_shared_types::ff_field;
use remainder_shared_types::pedersen::{CommittedScalar, CommittedVector, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;
use remainder_shared_types::Field;
use remainder_shared_types::{
    config::{global_config::global_claim_agg_strategy, ClaimAggregationStrategy},
    curves::PrimeOrderCurve,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
/// This struct represents what a proof looks like for one layer of GKR, but Hyrax version.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxLayerProof<C: PrimeOrderCurve> {
    /// This is the proof of the sumcheck rounds for that layer.
    pub proof_of_sumcheck: ProofOfSumcheck<C>,
    /// The values of the claims made by this layer, in commitment form.
    pub commitments: Vec<C>,
    /// This is the associated proof of products for the final commitments of that layer.
    pub proofs_of_product: Vec<ProofOfProduct<C>>,
    /// This is the proof of claim aggregation for the associated claims that were
    /// aggregated to make a claim on this layer.
    /// Is None if we are using the RLC method of claim aggregation. Populated otherwise.
    pub maybe_proof_of_claim_agg: Option<ProofOfClaimAggregation<C>>,
}

impl<C: PrimeOrderCurve> HyraxLayerProof<C> {
    /// Generate the associated \alpha_i commitment for round i. This is
    /// essentially the coefficients to the univariate committed to (using the appropriate
    /// generators for that round, i.e. accounting for the zero padding of the sumcheck messages).
    #[allow(clippy::too_many_arguments)]
    fn commit_to_round(
        underlying_layer: &mut LayerEnum<C::Scalar>,
        committer: &PedersenCommitter<C>,
        bit_index: usize,
        round_number: usize,
        max_degree: usize,
        num_rounds: usize,
        blinding_rng: &mut impl Rng,
        random_coefficients: &[C::Scalar],
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> CommittedVector<C> {
        // The univariate evaluations for that round (i.e. f(0), f(1), ...)
        let round_evaluations = underlying_layer
            .compute_round_sumcheck_message(bit_index, random_coefficients)
            .unwrap();

        // Convert the evaluations above into coefficients (in Hyrax, it is the coefficients, not
        // the evaluations, that are used)
        // Initial padding for the beginning 0s.
        let mut round_coefficients = vec![C::Scalar::ZERO; (max_degree + 1) * round_number];
        // Pad the coefficients to form the $v^{(`round_index`)}$ vector, so that only the generators for this round are used
        let computed_coeffs = converter.convert_to_coefficients(round_evaluations);
        round_coefficients.extend(&computed_coeffs);
        // Padding the current round until it is the max_degree size.
        round_coefficients.extend(vec![
            C::Scalar::ZERO;
            (max_degree + 1) - computed_coeffs.len()
        ]);
        // Padding the entire vector to be the size of (`max_degree` + 1) * `num_rounds`
        round_coefficients.extend(vec![
            C::Scalar::ZERO;
            (max_degree + 1) * (num_rounds - round_number - 1)
        ]);
        let blinding_factor = C::Scalar::random(blinding_rng);

        committer.committed_vector(&round_coefficients, &blinding_factor)
    }

    #[allow(clippy::type_complexity)]
    /// Produce a [HyraxLayerProof] for a given layer, given the unaggregated claims on that layer.
    /// Return also a [HyraxClaim] representing the aggregated claim.
    pub fn prove(
        // The layer that we are proving
        layer: &mut LayerEnum<C::Scalar>,
        // The claims on that layer (unaggregated)
        claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        // The output MLEs from this layer, whose bookkeeping tables combined make the layerwise bookkeeping table.
        output_mles_from_layer: Vec<DenseMle<C::Scalar>>,
        committer: &PedersenCommitter<C>,
        mut blinding_rng: &mut (impl CryptoRng + RngCore),
        transcript: &mut impl ECTranscriptTrait<C>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> (Self, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>) {
        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                vec![C::Scalar::ONE]
            }
            ClaimAggregationStrategy::RLC => {
                transcript.get_scalar_field_challenges("RLC Claim Agg Coefficients", claims.len())
            }
        };
        let (maybe_proof_of_claim_agg, claim_points, claimed_eval) =
            if let &mut LayerEnum::MatMult(_) = layer {
                let interpolant_coeffs = if claims.len() > 1 {
                    // NB we don't use aggregate_claims here because the sampling of the evaluation
                    // point for the aggregate claim needs to happen elsewhere in Hyrax.
                    // Convert to a ClaimGroup so that we can use the helper functions
                    let claim_group = ClaimGroup::new_from_raw_claims(
                        claims
                            .iter()
                            .map(|hyrax_claim| hyrax_claim.to_raw_claim())
                            .collect_vec(),
                    )
                    .unwrap();
                    // Calculate the evaluations at 0, 1, 2, ..
                    let wlx_evals = get_wlx_evaluations(
                        claim_group.get_claim_points_matrix(),
                        claim_group.get_results(),
                        get_indexed_layer_mles_to_combine(output_mles_from_layer),
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
                    claims,
                    &interpolant_coeffs,
                    committer,
                    &mut blinding_rng,
                    transcript,
                );
                layer.initialize(&agg_claim.point).unwrap();
                (
                    Some(proof_of_claim_agg),
                    &vec![agg_claim.point],
                    agg_claim.evaluation,
                )
            } else {
                match global_claim_agg_strategy() {
                    ClaimAggregationStrategy::Interpolative => {
                        let interpolant_coeffs = if claims.len() > 1 {
                            // NB we don't use aggregate_claims here because the sampling of the evaluation
                            // point for the aggregate claim needs to happen elsewhere in Hyrax.
                            // Convert to a ClaimGroup so that we can use the helper functions
                            let claim_group = ClaimGroup::new_from_raw_claims(
                                claims
                                    .iter()
                                    .map(|hyrax_claim| hyrax_claim.to_raw_claim())
                                    .collect_vec(),
                            )
                            .unwrap();
                            // Calculate the evaluations at 0, 1, 2, ..
                            let wlx_evals = get_wlx_evaluations(
                                claim_group.get_claim_points_matrix(),
                                claim_group.get_results(),
                                get_indexed_layer_mles_to_combine(output_mles_from_layer),
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
                            claims,
                            &interpolant_coeffs,
                            committer,
                            &mut blinding_rng,
                            transcript,
                        );
                        layer.initialize(&agg_claim.point).unwrap();
                        (
                            Some(proof_of_claim_agg),
                            &vec![agg_claim.point],
                            agg_claim.evaluation,
                        )
                    }
                    ClaimAggregationStrategy::RLC => {
                        let rlc_eval = claims.iter().zip(random_coefficients.iter()).fold(
                            CommittedScalar::zero(),
                            |acc, (elem, random_coeff)| {
                                acc + elem.evaluation.clone() * *random_coeff
                            },
                        );
                        let raw_claims = claims
                            .iter()
                            .map(|claim| claim.to_raw_claim())
                            .collect_vec();
                        layer
                            .initialize_rlc(&random_coefficients, &raw_claims.iter().collect_vec());
                        (
                            None,
                            &claims.iter().map(|claim| claim.point.clone()).collect_vec(),
                            rlc_eval,
                        )
                    }
                }
            };

        // These are going to be the commitments to the sumcheck messages.
        let mut messages: Vec<CommittedVector<C>> = vec![];
        // These are the challenges for this layer of sumcheck. Append as we go.
        let mut bindings: Vec<C::Scalar> = vec![];

        // Note that the commitment to the aggregate evaluationp `eval` does not need to be added to the
        // transcript since it is derived from commitments that are added to the transcript already
        // (the commitments to the coefficients of the interpolant).
        let degree = layer.max_degree();
        // The number of sumcheck rounds for this layer
        let sumcheck_round_indices = layer.sumcheck_round_indices();
        let num_rounds = sumcheck_round_indices.len();

        // Go through each of the sumcheck rounds and produce the \alpha_i messages.
        sumcheck_round_indices
            .iter()
            .enumerate()
            .for_each(|(round_number, bit_idx)| {
                let round_commit = HyraxLayerProof::commit_to_round(
                    layer,
                    committer,
                    *bit_idx,
                    round_number,
                    degree,
                    num_rounds,
                    &mut blinding_rng,
                    &random_coefficients,
                    converter,
                );
                messages.push(round_commit.clone());
                transcript
                    .append_ec_point("Commitment to sumcheck message", round_commit.commitment);

                let challenge = transcript.get_scalar_field_challenge("sumcheck round challenge");
                bindings.push(challenge);
                layer.bind_round_variable(*bit_idx, challenge).unwrap();
            });

        // Get the post sumcheck layer
        let post_sumcheck_layer = layer.get_post_sumcheck_layer(
            &bindings,
            &claim_points
                .iter()
                .map(|claim_point| claim_point.as_slice())
                .collect_vec(),
            &random_coefficients,
        );

        // Commit to all necessary values
        let mut post_sumcheck_layer_committed =
            commit_to_post_sumcheck_layer(&post_sumcheck_layer, committer, &mut blinding_rng);
        post_sumcheck_layer_committed.remove_add_values(false);

        // Get the commitments (i.e. points on C)
        let commitments =
            committed_scalar_psl_as_commitments(&post_sumcheck_layer_committed).get_values();

        // Add each of the commitments to the transcript
        transcript.append_ec_points(
            "Commitments to all the layer's leaf values and intermediates",
            &commitments,
        );

        // Get the claims made in this layer
        let committed_claims = get_all_claims(&post_sumcheck_layer_committed);

        // Proof of sumcheck
        // Note that product_evaluations have already been added to the transcript (along with the rest of the commitments)
        let proof_of_sumcheck = ProofOfSumcheck::prove(
            &claimed_eval,
            &messages,
            degree,
            &post_sumcheck_layer_committed,
            &bindings,
            committer,
            blinding_rng,
            transcript,
        );

        // perform the proof of products
        let proofs_of_product = post_sumcheck_layer_committed
            .get_product_triples()
            .iter()
            .map(|(x, y, z)| {
                ProofOfProduct::prove(&x, &y, &z, committer, &mut blinding_rng, transcript)
            })
            .collect();

        (
            HyraxLayerProof {
                proof_of_sumcheck,
                commitments,
                proofs_of_product,
                maybe_proof_of_claim_agg,
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
        layer_desc: &impl LayerDescription<C::Scalar>,
        // commitments to the unaggregated claims
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Vec<HyraxClaim<C::Scalar, C>> {
        let HyraxLayerProof {
            maybe_proof_of_claim_agg,
            commitments,
            proofs_of_product,
            proof_of_sumcheck,
        } = proof;

        let random_coefficients = match global_claim_agg_strategy() {
            ClaimAggregationStrategy::Interpolative => {
                vec![C::Scalar::ONE]
            }
            ClaimAggregationStrategy::RLC => transcript
                .get_scalar_field_challenges("RLC Claim Agg Coefficients", claim_commitments.len()),
        };

        // Verify the proof of claim aggregation
        let (claim_points, claim_eval) = if let Some(proof_of_claim_agg) = maybe_proof_of_claim_agg
        {
            let agg_claim = proof_of_claim_agg.verify(claim_commitments, committer, transcript);
            (&vec![agg_claim.point], agg_claim.evaluation)
        } else {
            let rlc_eval = claim_commitments
                .iter()
                .zip(random_coefficients.iter())
                .fold(C::zero(), |acc, (elem, random_coeff)| {
                    acc + elem.evaluation * *random_coeff
                });
            (
                &claim_commitments
                    .iter()
                    .map(|claim_commit| claim_commit.point.clone())
                    .collect_vec(),
                rlc_eval,
            )
        };

        // The number of sumcheck rounds w.r.t. to the beta table rather than just the expression.
        // Because the beta table number of variables is exactly the number of points in the claim
        // made on that layer, we take the max of the number of variables in the expression and
        // the number of variables in the beta table.
        let num_sumcheck_rounds_expected = layer_desc.sumcheck_round_indices().len();

        // Verify the proof of sumcheck
        // Append first sumcheck message to transcript, which is the proported sum.
        if num_sumcheck_rounds_expected > 0 {
            transcript.append_ec_point(
                "Commitment to sumcheck message",
                proof_of_sumcheck.messages[0],
            );
        }

        // Collect the "bindings" for each of the sumcheck rounds. Add sumcheck messages to transcript.
        let mut bindings: Vec<C::Scalar> = vec![];
        proof_of_sumcheck
            .messages
            .iter()
            .skip(1)
            .for_each(|message| {
                let challenge = transcript.get_scalar_field_challenge("sumcheck round challenge");
                bindings.push(challenge);

                transcript.append_ec_point("Commitment to sumcheck message", *message);
            });

        // Final challenge in sumcheck -- needed for "oracle query".
        if num_sumcheck_rounds_expected > 0 {
            let final_chal = transcript.get_scalar_field_challenge("sumcheck round challenge");
            bindings.push(final_chal);
        }

        // Verify that we have the correct number of bindings
        assert_eq!(bindings.len(), num_sumcheck_rounds_expected);

        // Add the commitments made by the prover to the transcript
        transcript.append_ec_points(
            "Commitments to all the layer's leaf values and intermediates",
            commitments,
        );

        let mut post_sumcheck_layer_desc = layer_desc.get_post_sumcheck_layer(
            &bindings,
            &claim_points
                .iter()
                .map(|claim_point| claim_point.as_slice())
                .collect_vec(),
            &random_coefficients,
        );

        // Avoid committing to unnecessary values
        post_sumcheck_layer_desc.remove_add_values(false);
        let post_sumcheck_layer: PostSumcheckLayerTree<C::Scalar, C> =
            new_with_values(&post_sumcheck_layer_desc, commitments, &mut 0);
        
        // println!("\n\nVERIFIER_LAYER: {:?}", post_sumcheck_layer);

        // Verify the proof of sumcheck!
        proof_of_sumcheck.verify(
            &claim_eval,
            layer_desc.max_degree(),
            &post_sumcheck_layer,
            &bindings,
            committer,
            transcript,
        );

        // Extract the triples of commitments that must be proven in the proof of product
        // and verify the proofs of product
        let product_triples: Vec<(C, C, C)> = post_sumcheck_layer.get_product_triples();
        assert_eq!(product_triples.len(), proofs_of_product.len());
        product_triples
            .iter()
            .zip(proofs_of_product.iter())
            .for_each(|((x, y, z), proof)| {
                proof.verify(*x, *y, *z, committer, transcript);
            });
        println!("BBB");

        // Extract the claims that the prover implicitly made on other layers by sending `commitments`.
        get_all_claims(&post_sumcheck_layer)
    }
}

// ---------- This is where all the Hyrax [PostSumcheckLayerTree]-specific stuff is going! ----------
/// Evaluate the [PostSumcheckLayerTree] to a single CommittedScalar.
pub fn evaluate_committed_scalar<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayerTree<C::Scalar, CommittedScalar<C>>,
    one: CommittedScalar<C>,
) -> CommittedScalar<C> {
    post_sumcheck_layer.get_result(one)
}

/// Turn all the CommittedScalars into commitments i.e. Cs.
/// (This can't be a From implementation, since PostSumcheckLayer is not from this crate).
pub fn committed_scalar_psl_as_commitments<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayerTree<C::Scalar, CommittedScalar<C>>,
) -> PostSumcheckLayerTree<C::Scalar, C> {
    match post_sumcheck_layer {
        PostSumcheckLayerTree::Constant { coefficient } => PostSumcheckLayerTree::Constant { 
            coefficient: coefficient.clone()
        },
        PostSumcheckLayerTree::Mle { layer_id, point, value } => PostSumcheckLayerTree::Mle { 
            layer_id: *layer_id,
            point: point.clone(),
            value: value.commitment,
        },
        PostSumcheckLayerTree::Add { left, right, value } => PostSumcheckLayerTree::Add { 
            left: Box::new(committed_scalar_psl_as_commitments(left)),
            right: Box::new(committed_scalar_psl_as_commitments(right)),
            value: if let Some(val) = value {
                Some(val.commitment)
            } else {
                None
            }
        },
        PostSumcheckLayerTree::Mult { left, right, value } => PostSumcheckLayerTree::Mult { 
            left: Box::new(committed_scalar_psl_as_commitments(left)),
            right: Box::new(committed_scalar_psl_as_commitments(right)),
            value: if let Some(val) = value {
                Some(val.commitment)
            } else {
                None
            }
        },
    }
}

/// Evaluate the PostSumcheckLayer to a single scalar
pub fn evaluate_committed_psl<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayerTree<C::Scalar, C>,
    one: C,
) -> C {
    post_sumcheck_layer.get_result(one)
}

/// Obtained all the claims in a [PostSumcheckLayerTree] in post-order
fn get_all_claims<F: Field, T: Clone + Serialize + for<'de> Deserialize<'de>>(
    tree: &PostSumcheckLayerTree<F, T>
) -> Vec<HyraxClaim<F, T>> {
    match tree {
        PostSumcheckLayerTree::Constant { .. } => Vec::new(),
        PostSumcheckLayerTree::Mle { layer_id, point, value } => {
            vec![HyraxClaim {
                to_layer_id: *layer_id,
                point: point.clone(),
                evaluation: value.clone(),
            }]
        }
        PostSumcheckLayerTree::Add { left, right, .. } => {
            let left_claims = get_all_claims(left);
            let right_claims = get_all_claims(right);
            left_claims.into_iter().chain(right_claims).collect()
        }
        PostSumcheckLayerTree::Mult { left, right, .. } => {
            let left_claims = get_all_claims(left);
            let right_claims = get_all_claims(right);
            left_claims.into_iter().chain(right_claims).collect()
        }
    }
}

/// Implementation of HyraxClaim as used by the prover
impl<C: PrimeOrderCurve> HyraxClaim<C::Scalar, CommittedScalar<C>> {
    /// Form a new [HyraxClaim] given a challenge point and [CommittedScalar].
    ///
    /// NOTE: Only to be called during testing.
    pub fn new_raw(point: Vec<C::Scalar>, eval: CommittedScalar<C>) -> Self {
        Self {
            point,
            to_layer_id: LayerId::Input(0),
            evaluation: eval,
        }
    }
    /// Convert to a [RawClaim] for claim aggregation
    pub fn to_raw_claim(&self) -> RawClaim<C::Scalar> {
        RawClaim::new(self.point.clone(), self.evaluation.value)
    }

    /// Convert to a HyraxClaim<C::Scalar, C>
    pub fn to_claim_commitment(&self) -> HyraxClaim<C::Scalar, C> {
        HyraxClaim {
            point: self.point.clone(),
            to_layer_id: self.to_layer_id,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Field, T: DeserializeOwned")]
pub struct HyraxClaim<F: Field, T: Serialize + DeserializeOwned> {
    /// Id of the layer upon which the claim is made
    pub to_layer_id: LayerId,
    /// The evaluation point
    pub point: Vec<F>,
    /// The value of the claim
    pub evaluation: T,
}

/// Returns a CommittedScalar version of the PostSumcheckLayer.
/// Note that we cannot simply commit to every scalar evaluation,
/// instead, depending on whether the entry is public or private, 
/// there are different types of operations
/// Corresponds to [PostSumcheckLayerTree::get_result]
pub fn commit_to_post_sumcheck_layer<C: PrimeOrderCurve>(
    post_sumcheck_layer: &PostSumcheckLayerTree<C::Scalar, C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut blinding_rng: &mut impl Rng,
) -> PostSumcheckLayerTree<C::Scalar, CommittedScalar<C>> {
    let comm = PostSumcheckLayerTree::<C::Scalar, CommittedScalar<C>>::commit(
        post_sumcheck_layer,
        &mut |value| committer.committed_scalar(&value, &C::Scalar::random(&mut blinding_rng)),
        committer.committed_scalar(&C::Scalar::ONE, &C::Scalar::ZERO),
    );
    comm.0
}
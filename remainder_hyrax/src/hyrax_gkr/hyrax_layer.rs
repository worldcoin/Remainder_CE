use crate::hyrax_primitives::proof_of_equality::ProofOfEquality;
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
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::halo2curves::group::ff::Field;
use remainder_shared_types::transcript::Transcript;
use std::{collections::HashMap, marker::PhantomData};

/// This struct represents what a proof looks like for one layer of GKR, but Hyrax version.
pub struct HyraxLayerProof<C: PrimeOrderCurve, Tr: Transcript<C::Scalar, C::Base>> {
    /// This is the proof of the sumcheck rounds for that layer.
    pub proof_of_sumcheck: ProofOfSumcheck<C>,
    /// The values of the claims made by this layer, in commitment form.
    pub commitments: Vec<C>,
    /// This is the associated proof of products for the final commitments of that layer.
    pub proofs_of_product: Vec<ProofOfProduct<C>>,
    /// This is the proof of claim aggregation for the associated claims that were
    /// aggregated to make a claim on this layer.
    pub proof_of_claim_agg: ProofOfClaimAggregation<C>,
    _marker: PhantomData<Tr>,
}

impl<C: PrimeOrderCurve, Tr: Transcript<C::Scalar, C::Base>> HyraxLayerProof<C, Tr> {
    /// Generate the associated \alpha_i commitment for round i. This is
    /// essentially the coefficients to the univariate committed to (using the appropriate
    /// generators for that round, i.e. accounting for the zero padding of the sumcheck messages).
    fn commit_to_round(
        underlying_layer: &mut LayerEnum<C::Scalar, C::Base, Tr>,
        committer: &PedersenCommitter<C>,
        round_index: usize,
        max_degree: usize,
        challenge: C::Scalar,
        num_rounds: usize,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> CommittedVector<C> {
        // The univariate evaluations for that round (i.e. f(0), f(1), ...)
        let round_evaluations = match underlying_layer {
            LayerEnum::Gkr(ref mut layer) => layer.prove_round(round_index, challenge).unwrap(),
            LayerEnum::EmptyLayer(ref mut layer) => vec![gather_combine_all_evals(&layer.expr)
                .map_err(LayerError::ExpressionError)
                .unwrap()],
            LayerEnum::IdentityGate(ref mut layer) => {
                let mle_refs = layer
                    .phase_1_mles
                    .as_mut()
                    .ok_or(GateError::Phase1InitError)
                    .unwrap();
                prove_round_identity(round_index, challenge, mle_refs).unwrap()
            }
            LayerEnum::MatMult(ref mut layer) => {
                let matrix_a = &mut layer.matrix_a;
                let matrix_b = &mut layer.matrix_b;
                prove_round_matmul(matrix_a, matrix_b, round_index, challenge)
            }
        };

        // Convert the evaluations above into coefficients (in Hyrax, it is the coefficients, not
        // the evaluations, that are used)
        let mut round_coefficients = vec![C::Scalar::zero(); (max_degree + 1) * round_index];
        // Pad the coefficients to form the $v^{(`round_index`)}$ vector, so that only the generators for this round are used
        let computed_coeffs = converter.convert_to_coefficients(round_evaluations);
        round_coefficients.extend(&computed_coeffs);
        round_coefficients.extend(vec![
            C::Scalar::zero();
            (max_degree + 1) - computed_coeffs.len()
        ]);
        round_coefficients.extend(vec![
            C::Scalar::zero();
            (max_degree + 1) * (num_rounds - round_index - 1)
        ]);
        let blinding_factor = C::Scalar::random(blinding_rng);
        let commitment = committer.committed_vector(&round_coefficients, &blinding_factor);
        commitment
    }

    /// A function to commit to the first round of sumcheck because this is done slightly differently, where we need
    /// an input to the claim on that layer along with the challenges.
    fn commit_to_first_round_of_sumcheck(
        underlying_layer: &mut LayerEnum<C::Scalar, C::Base, Tr>,
        claim_challenges: &[C::Scalar],
        max_degree: usize,
        committer: &PedersenCommitter<C>,
        num_rounds: usize,
        blinding_rng: &mut impl Rng,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> CommittedVector<C> {
        let round_evaluations = match underlying_layer {
            LayerEnum::Gkr(ref mut layer) => layer.start_sumcheck(claim_challenges).unwrap().0,
            LayerEnum::EmptyLayer(ref mut layer) => layer.first_message(),
            LayerEnum::IdentityGate(ref mut layer) => layer.init_phase_1(claim_challenges).unwrap(),
            LayerEnum::MatMult(ref mut layer) => layer.pre_processing_step(claim_challenges),
        };
        let mut round_coefficients = converter.convert_to_coefficients(round_evaluations);
        round_coefficients.extend(vec![
            C::Scalar::zero();
            (max_degree + 1) - round_coefficients.len()
        ]);

        if num_rounds > 0 {
            round_coefficients.extend(vec![C::Scalar::zero(); (max_degree + 1) * (num_rounds - 1)]);
        }

        let blinding_factor = C::Scalar::random(blinding_rng);
        let commitment = committer.committed_vector(&round_coefficients, &blinding_factor);
        commitment
    }

    /// Produce a [HyraxLayerProof] for a given layer, given the unaggregated claims on that layer.
    /// Return also a [HyraxClaim] representing the aggregated claim.
    pub fn prove(
        // The layer that we are proving
        mut layer: &mut LayerEnum<C::Scalar, C::Base, Tr>,
        // The claims on that layer (unaggregated)
        claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        mut blinding_rng: &mut impl Rng,
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> (Self, Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>) {
        let interpolant_coeffs = if claims.len() > 1 {
            // CALCULATE THE COEFFICIENTS OF THE POLYNOMIAL THAT INTERPOLATES THE CLAIMS ON THIS LAYER
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
        // Note that the commitment to the aggregate evaluation `eval` does not need to be added to the
        // transcript since it is derived from commitments that are added to the transcript already
        // (the commitments to the coefficients of the interpolant).

        let degree = layer.max_degree();
        // The number of sumcheck rounds w.r.t. to the beta table rather than just the expression.
        // Because the beta table number of variables is exactly the number of points in the claim
        // made on that layer, we take the max of the number of variables in the expression and
        // the number of variables in the beta table.
        let num_rounds = match layer {
            LayerEnum::EmptyLayer(_) | LayerEnum::Gkr(_) => {
                std::cmp::max(layer.layer_size(), agg_claim.point.len())
            }
            _ => layer.layer_size(),
        };
        // These are going to be the commitments to the sumcheck messages.
        let mut messages: Vec<CommittedVector<C>> = vec![];
        // These are the challenges for this layer of sumcheck. Append as we go.
        let mut bindings: Vec<C::Scalar> = vec![];

        // \alpha_0 message.
        if num_rounds > 0 {
            let first_sumcheck_message = HyraxLayerProof::commit_to_first_round_of_sumcheck(
                &mut layer,
                &agg_claim.point,
                degree,
                &committer,
                num_rounds,
                &mut blinding_rng,
                converter,
            );

            append_x_y_to_transcript_single(
                &first_sumcheck_message.commitment,
                transcript,
                "first sumcheck message commitment x",
                "first sumcheck message commitment y",
            );
            messages.push(first_sumcheck_message);
        } else {
            match layer {
                LayerEnum::EmptyLayer(layer) => {
                    layer.expr.index_mle_indices(0);
                }
                _ => {
                    panic!("only empty layer should have 0 rounds")
                }
            }
        }

        // Go through each of the sumcheck rounds and produce the \alpha_i messages.
        (1..num_rounds).for_each(|round| {
            let challenge = transcript
                .get_scalar_field_challenge("sumcheck round challenge")
                .unwrap();
            bindings.push(challenge);

            let round_commit = HyraxLayerProof::commit_to_round(
                &mut layer,
                &committer,
                round,
                degree,
                challenge,
                num_rounds,
                &mut blinding_rng,
                converter,
            );
            append_x_y_to_transcript_single(
                &round_commit.commitment,
                transcript,
                "sumcheck message commitment x",
                "sumcheck message commitment y",
            );
            messages.push(round_commit);
        });

        if num_rounds > 0 {
            // Final round of sumcheck -- only if we have more than 0 rounds
            let final_challenge = transcript
                .get_scalar_field_challenge("final round challenge")
                .unwrap();
            bindings.push(final_challenge);
            layer.fix_final_round(final_challenge, num_rounds);
        }

        // Get the post sumcheck layer
        let post_sumcheck_layer = layer.get_post_sumcheck_layer(&bindings, &agg_claim.point);

        // Commit to all the necessary values
        let post_sumcheck_layer_committed =
            commit_to_post_sumcheck_layer(&post_sumcheck_layer, &committer, &mut blinding_rng);

        // Get the commitments (i.e. points on C)
        let commitments = post_sumcheck_layer_committed.as_commitments().get_values();

        // Add each of the commitments to the transcript
        commitments.iter().for_each(|commitment| {
            append_x_y_to_transcript_single(
                &commitment.clone(),
                transcript,
                "commit to product input or output x",
                "commit to product input or output y",
            );
        });

        // Get the claims made in this layer
        let committed_claims: Vec<_> = post_sumcheck_layer_committed
            .0
            .iter()
            .map(|product| product.get_claims())
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
                _marker: PhantomData,
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
        proof: &HyraxLayerProof<C, Tr>,
        // a description of the layer being proven
        layer_desc: &LayerDescription<C::Scalar>,
        // commitments to the unaggregated claims
        claim_commitments: &Vec<HyraxClaim<C::Scalar, C>>,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
    ) -> Vec<HyraxClaim<C::Scalar, C>> {
        let HyraxLayerProof {
            proof_of_claim_agg,
            commitments,
            proofs_of_product,
            proof_of_sumcheck,
            _marker,
        } = proof;

        // Verify the proof of claim aggregation
        let agg_claim = proof_of_claim_agg.verify(&claim_commitments, &committer, transcript);

        // The number of sumcheck rounds w.r.t. to the beta table rather than just the expression.
        // Because the beta table number of variables is exactly the number of points in the claim
        // made on that layer, we take the max of the number of variables in the expression and
        // the number of variables in the beta table.
        let num_sumcheck_rounds_expected = match layer_desc.layer_desc_enum {
            LayerDescEnum::EmptyLayer(_) | LayerDescEnum::Gkr(_) => {
                std::cmp::max(layer_desc.layer_size, agg_claim.point.len())
            }
            _ => layer_desc.layer_size,
        };

        // Verify the proof of sumcheck
        // Add first sumcheck message to transcript, which is the proported sum.
        if num_sumcheck_rounds_expected > 0 {
            append_x_y_to_transcript_single(
                &proof_of_sumcheck.messages[0],
                transcript,
                "first sumcheck message commitment x",
                "first sumcheck message commitment y",
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
                append_x_y_to_transcript_single(
                    &message as &C,
                    transcript,
                    "sumcheck message commitment x",
                    "sumcheck message commitment y",
                );
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
        commitments.iter().for_each(|commitment| {
            append_x_y_to_transcript_single(
                &commitment.clone(),
                transcript,
                "commit to product input or output x",
                "commit to product input or output y",
            );
        });

        // Build the PostSumcheckLayer from the commitments and the layer description
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
            .map(|product| product.get_claims())
            .flatten()
            .collect_vec();

        claims
    }
}

// ---------- This is where all the Hyrax [PostSumcheckLayer]-specific stuff is going! ----------

impl<C: PrimeOrderCurve> PostSumcheckLayer<C::Scalar, CommittedScalar<C>> {
    /// Evaluate the PostSumcheckLayer to a single CommittedScalar.
    pub fn evaluate_committed_scalar(&self) -> CommittedScalar<C> {
        self.0
            .iter()
            .fold(CommittedScalar::zero(), |acc, (product)| {
                acc + product.get_result() * product.coefficient
            })
    }

    /// Turn all the CommittedScalars into commitments i.e. Cs.
    pub fn as_commitments(&self) -> PostSumcheckLayer<C::Scalar, C> {
        PostSumcheckLayer(
            self.0
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
}

impl<C: PrimeOrderCurve> PostSumcheckLayer<C::Scalar, C> {
    /// Evaluate the PostSumcheckLayer to a single scalar
    pub fn evaluate(&self) -> C {
        self.0.iter().fold(C::zero(), |acc, product| {
            acc + product.get_result() * product.coefficient
        })
    }
}

impl<F: FieldExt, T: Clone> Product<F, T> {
    /// Return the claims made on other layers by the atomic factors of this product.
    pub fn get_claims(&self) -> Vec<HyraxClaim<F, T>> {
        self.intermediates
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
            .map(|product| commit_to_product(product, committer, &mut blinding_rng))
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

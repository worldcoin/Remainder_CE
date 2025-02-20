use std::collections::HashMap;

use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use rand::{CryptoRng, Rng, RngCore};
use remainder::{
    input_layer::InputLayerDescription, layer::LayerId, mle::evals::MultilinearExtension,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    pedersen::{CommittedScalar, PedersenCommitter},
    transcript::ec_transcript::ECTranscriptTrait,
};
use remainder_shared_types::{ff_field, Zeroizable};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::hyrax_pcs::HyraxPCSEvaluationProof;

use super::hyrax_layer::HyraxClaim;

/// The proof structure for a Hyrax input layer.
/// Proves multiple claims without claim aggregation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxInputLayerProof<C: PrimeOrderCurve> {
    /// The ID of the layer that this is a proof for
    pub layer_id: LayerId,
    /// The commitment to the input polynomial
    pub input_commitment: Vec<C>,
    /// The evaluation points and evaluation proofs for the input layer claims, sorted by evaluation point.
    pub evaluation_proofs: Vec<HyraxPCSEvaluationProof<C>>,
}

impl<C: PrimeOrderCurve> HyraxInputLayerProof<C> {
    /// Create a proof of the claims on a Hyrax input layer.
    /// # Arguments:
    /// * `input_layer_desc` - The description of the input layer
    /// * `prover_commitment` - The commitment to the input layer
    /// * `committed_claims` - The claims made on this layer (in any order)
    pub fn prove(
        input_layer_desc: &HyraxInputLayerDescription,
        prover_commitment: &mut HyraxProverInputCommitment<C>,
        committed_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut (impl CryptoRng + RngCore),
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Self {
        let eval_proof_timer = start_timer!(|| "eval proof timer");
        // Sort the claims by evaluation point
        let mut committed_claims = committed_claims.to_vec();
        committed_claims.sort_by(|a, b| a.point.cmp(&b.point));
        let claims_grouped_by_common_points =
            group_claims_by_common_points_with_dimension::<C, CommittedScalar<C>>(
                committed_claims,
                input_layer_desc.log_num_cols,
            );
        let evaluation_proofs = claims_grouped_by_common_points
            .iter()
            .map(|claim_group| {
                HyraxPCSEvaluationProof::prove(
                    input_layer_desc.log_num_cols,
                    &prover_commitment.mle,
                    claim_group,
                    committer,
                    blinding_rng,
                    transcript,
                    &mut prover_commitment.blinding_factors_matrix,
                )
            })
            .collect_vec();
        // Zeroize each of the blinding factors once we have committed to all of the claims.
        prover_commitment
            .blinding_factors_matrix
            .iter_mut()
            .for_each(|blinding_factor| {
                blinding_factor.zeroize();
            });
        end_timer!(eval_proof_timer);

        HyraxInputLayerProof {
            layer_id: input_layer_desc.layer_id,
            input_commitment: prover_commitment.commitment.clone(),
            evaluation_proofs,
        }
    }

    /// Verify a Hyrax Input Layer proof, establishing all the provided claims on the input layer.
    /// # Arguments:
    /// * `input_layer_desc` - The description of the input layer
    /// * `claim_commitments` - The commitments to the claims, in any order
    pub fn verify(
        &self,
        input_layer_desc: &HyraxInputLayerDescription,
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // Sort the claims by evaluation point
        let mut claim_commitments = claim_commitments.to_vec();
        claim_commitments.sort_by(|a, b| a.point.cmp(&b.point));
        let claims_grouped_by_common_points = group_claims_by_common_points_with_dimension::<C, C>(
            claim_commitments,
            input_layer_desc.log_num_cols,
        );

        // Check there are the same number of claims as evaluation proofs
        assert_eq!(
            self.evaluation_proofs.len(),
            claims_grouped_by_common_points.len()
        );

        // Verify each evaluation proof
        claims_grouped_by_common_points
            .iter()
            .zip(&self.evaluation_proofs)
            .for_each(|(claim_group, eval_proof)| {
                assert!(claim_group
                    .iter()
                    .all(|claim| claim.point.len() == input_layer_desc.num_vars));
                // assert!(claim_group
                //     .iter()
                //     .all(|claim| claim.evaluation == eval_proof.commitment_to_evaluation));
                eval_proof.verify(
                    input_layer_desc.log_num_cols,
                    committer,
                    &self.input_commitment,
                    &claim_group.iter().map(|claim| &claim.point).collect_vec(),
                    transcript,
                );
            });
    }
}

fn group_claims_by_common_points_with_dimension<
    C: PrimeOrderCurve,
    T: Clone + Serialize + for<'de> Deserialize<'de>,
>(
    claims: Vec<HyraxClaim<C::Scalar, T>>,
    log_n_cols: usize,
) -> Vec<Vec<HyraxClaim<C::Scalar, T>>> {
    let mut claim_groups: Vec<Vec<HyraxClaim<C::Scalar, T>>> = Vec::new();
    for claim in claims.into_iter() {
        let mut maybe_inserted_claim = Some(claim);
        for claim_group in claim_groups.iter_mut() {
            if !claim_group.is_empty() {
                if let Some(ref claim) = maybe_inserted_claim {
                    if claim_group[0].point[log_n_cols..] == claim.point[log_n_cols..] {
                        claim_group.push(maybe_inserted_claim.take().unwrap());
                        break;
                    }
                }
            }
        }
        if let Some(claim) = maybe_inserted_claim {
            claim_groups.push(vec![claim]);
        }
    }
    claim_groups
}

#[derive(Clone, Debug, PartialEq)]
/// The circuit description of a Hyrax input layer. Stores the shape information
/// of this layer. All of the functionality of Hyrax input layers are taken care
/// of in `remainder_hyrax/`, so this is meant just to generate a circuit
/// description.
pub struct HyraxInputLayerDescription {
    /// The input layer ID.
    pub layer_id: LayerId,
    /// The number of variables this Hyrax Input Layer is on.
    pub num_vars: usize,
    /// The log number of columns in the matrix form of the data that will be
    /// committed to in this input layer.
    pub log_num_cols: usize,
}

/// Type alias for a Hyrax input layer's description + optional precommit.
pub type HyraxInputLayerDescriptionWithPrecommit<C> = HashMap<
    LayerId,
    (
        HyraxInputLayerDescription,
        Option<HyraxProverInputCommitment<C>>,
    ),
>;

impl HyraxInputLayerDescription {
    /// Create a [HyraxInputLayerDescription] specifying the use of a square
    /// matrix ("default setup"; build the struct directly for custom setup).
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        let log_num_cols = num_bits / 2;
        Self {
            layer_id,
            num_vars: num_bits,
            log_num_cols,
        }
    }
}

impl From<InputLayerDescription> for HyraxInputLayerDescription {
    /// Convert an [InputLayerDescription] into a [HyraxInputLayerDescription]
    /// with a square matrix.
    fn from(input_layer_desc: InputLayerDescription) -> Self {
        HyraxInputLayerDescription::new(input_layer_desc.layer_id, input_layer_desc.num_vars)
    }
}

/// Given a [HyraxInputLayerDescription] and values for its MLE, compute the
/// [HyraxInputCommitment] for the input layer.
pub fn commit_to_input_values<C: PrimeOrderCurve>(
    input_layer_desc: &HyraxInputLayerDescription,
    input_mle: &MultilinearExtension<C::Scalar>,
    committer: &PedersenCommitter<C>,
    mut rng: &mut impl Rng,
) -> HyraxProverInputCommitment<C> {
    let num_rows = 1 << (input_layer_desc.num_vars - input_layer_desc.log_num_cols);
    // Sample the blinding factors
    let mut blinding_factors_matrix = vec![C::Scalar::ZERO; num_rows];
    blinding_factors_matrix
        .iter_mut()
        .take(num_rows)
        .for_each(|blinding_factor| {
            *blinding_factor = C::Scalar::random(&mut rng);
        });
    let mle_coeffs_vec = MultilinearExtension::new(input_mle.f.iter().collect_vec());
    let commitment_values = HyraxPCSEvaluationProof::compute_matrix_commitments(
        input_layer_desc.log_num_cols,
        &mle_coeffs_vec,
        committer,
        &blinding_factors_matrix,
    );
    HyraxProverInputCommitment {
        mle: mle_coeffs_vec,
        commitment: commitment_values,
        blinding_factors_matrix,
    }
}

/// The prover's view of the commitment to the input layer, which includes the
/// blinding factors and the plaintext values.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxProverInputCommitment<C: PrimeOrderCurve> {
    /// The plaintext values
    pub mle: MultilinearExtension<C::Scalar>,
    /// The verifier's view of the commitment
    pub commitment: Vec<C>,
    /// The blinding factors used in the commitment
    pub blinding_factors_matrix: Vec<C::Scalar>,
}

impl<C: PrimeOrderCurve> Zeroizable for HyraxProverInputCommitment<C> {
    fn zeroize(&mut self) {
        self.mle.zeroize();
        for c in &mut self.commitment {
            c.zeroize();
        }
        for bf in &mut self.blinding_factors_matrix {
            bf.zeroize();
        }
    }
}

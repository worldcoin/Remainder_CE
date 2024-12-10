use std::collections::HashMap;

use ark_std::{cfg_iter, end_timer, start_timer};
use itertools::Itertools;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use remainder::claims::claim_aggregation::get_wlx_evaluations;
use remainder::mle::evals::bit_packed_vector::num_bits;
use remainder::{
    claims::{claim_group::ClaimGroup, RawClaim},
    input_layer::InputLayerDescription,
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    pedersen::{CommittedScalar, PedersenCommitter},
    transcript::ec_transcript::ECTranscriptTrait,
};
use remainder_shared_types::{ff_field, Field};

use crate::{
    hyrax_pcs::HyraxPCSEvaluationProof,
    hyrax_primitives::{
        proof_of_claim_agg::ProofOfClaimAggregation, proof_of_equality::ProofOfEquality,
    },
    utils::vandermonde::VandermondeInverse,
};

use super::hyrax_layer::HyraxClaim;

/// The proof structure for a Hyrax input layer. Includes the
/// [ProofOfClaimAggregation], and the appropriate opening proof for opening the
/// polynomial commitment at a random evaluation point.
pub struct HyraxInputLayerProof<C: PrimeOrderCurve> {
    /// The ID of the layer that this is a proof for
    pub layer_id: LayerId,
    /// The proof of claim aggregation for this layer
    pub claim_agg_proof: ProofOfClaimAggregation<C>,
    /// The commitment to the input polynomial
    pub input_commitment: Vec<C>,
    /// The evaluation proof for the polynomial evaluated at a random point
    pub evaluation_proof: HyraxPCSEvaluationProof<C>,
    /// The proof of equality that the aggregated claim point and the commitment
    /// in the evaluation proof are indeed to the same value
    pub proof_of_equality: ProofOfEquality<C>,
}

impl<C: PrimeOrderCurve> HyraxInputLayerProof<C> {
    pub fn prove(
        input_layer_desc: &HyraxInputLayerDescription,
        prover_commitment: &HyraxProverInputCommitment<C>,
        committed_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        transcript: &mut impl ECTranscriptTrait<C>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> Self {
        // Calculate the coefficients of the polynomial that interpolates the
        // claims NB we don't use aggregate_claims here because the sampling of
        // the evaluation point for the aggregate claim needs to happen
        // elsewhere in Hyrax.
        let claims = ClaimGroup::new_from_raw_claims(
            committed_claims
                .iter()
                .map(|committed_claim| committed_claim.to_raw_claim())
                .collect_vec(),
        )
        .unwrap();
        let compute_vi_lx_eval_timer = start_timer!(|| "vilx evals for input");

        let wlx_evals = get_wlx_evaluations(
            claims.get_claim_points_matrix(),
            claims.get_results(),
            vec![DenseMle::new_from_multilinear_extension(
                prover_commitment.mle.clone(),
                input_layer_desc.layer_id,
                None,
                Some(0),
            )],
            claims.get_num_claims(),
            claims.get_num_vars(),
        )
        .unwrap();

        end_timer!(compute_vi_lx_eval_timer);

        let coeffs_timer = start_timer!(|| "convert to coeffs timer");
        let interpolant_coeffs = converter.convert_to_coefficients(wlx_evals);

        end_timer!(coeffs_timer);

        let claim_agg_timer = start_timer!(|| "claim agg input");
        let (proof_of_claim_agg, aggregated_claim): (
            ProofOfClaimAggregation<C>,
            HyraxClaim<C::Scalar, CommittedScalar<C>>,
        ) = ProofOfClaimAggregation::prove(
            committed_claims,
            &interpolant_coeffs,
            committer,
            blinding_rng,
            transcript,
        );
        end_timer!(claim_agg_timer);

        let eval_proof_timer = start_timer!(|| "eval proof timer");
        let evaluation_proof = HyraxPCSEvaluationProof::prove(
            input_layer_desc.log_num_cols,
            &prover_commitment.mle,
            &aggregated_claim.point,
            &aggregated_claim.evaluation.value,
            committer,
            blinding_rng,
            transcript,
            &prover_commitment.blinding_factors_matrix,
        );
        end_timer!(eval_proof_timer);

        let proof_of_equality = ProofOfEquality::prove(
            &aggregated_claim.evaluation,
            &evaluation_proof.commitment_to_evaluation,
            committer,
            blinding_rng,
            transcript,
        );

        HyraxInputLayerProof {
            layer_id: input_layer_desc.layer_id,
            input_commitment: prover_commitment.commitment.clone(),
            claim_agg_proof: proof_of_claim_agg,
            evaluation_proof,
            proof_of_equality,
        }
    }

    /// Verify a Hyrax Input Layer proof by verifying the inner proof of claim
    /// aggregation, and then verifying the opening proof by checking the claim.
    pub fn verify(
        &self,
        input_layer_desc: &HyraxInputLayerDescription,
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // Verify the proof of claim aggregation
        let agg_claim = self
            .claim_agg_proof
            .verify(claim_commitments, committer, transcript);

        // Verify the actual "evaluation" polynomial committed to at the random
        // point
        self.evaluation_proof.verify(
            input_layer_desc.log_num_cols,
            committer,
            &self.input_commitment,
            &agg_claim.point,
            transcript,
        );

        // Proof of equality for the aggregated claim and the evaluation proof
        // commitment
        self.proof_of_equality.verify(
            agg_claim.evaluation,
            self.evaluation_proof.commitment_to_evaluation.commitment,
            committer,
            transcript,
        );
    }
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
    pub num_bits: usize,
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
            num_bits,
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
    let num_rows = 1 << (input_layer_desc.num_bits - input_layer_desc.log_num_cols);
    // Sample the blinding factors
    let mut blinding_factors_matrix = vec![C::Scalar::ZERO; num_rows];
    blinding_factors_matrix
        .iter_mut()
        .take(num_rows)
        .for_each(|blinding_factor| {
            *blinding_factor = C::Scalar::random(&mut rng);
        });

    // We take the largest value in the MLE and compute the bits needed to
    // represent it.
    let input_mle_vec = input_mle.to_vec();
    let max_input_mle_value = cfg_iter!(input_mle_vec).max().unwrap();
    let max_num_bits_needed = num_bits(*max_input_mle_value);
    // If it is <= 128, then we can use an optimized version of the vector
    // commitment. Every element will be padded to fit to the next power of two
    // to the max number of bits needed.
    let maybe_optimized_num_bits = if max_num_bits_needed <= 128 {
        Some(max_num_bits_needed)
    } else {
        None
    };

    let mle_coeffs_vec = MultilinearExtension::new(input_mle.f.iter().collect_vec());
    let commitment_values = HyraxPCSEvaluationProof::compute_matrix_commitments(
        input_layer_desc.log_num_cols,
        &mle_coeffs_vec,
        committer,
        &blinding_factors_matrix,
        maybe_optimized_num_bits,
    );
    HyraxProverInputCommitment {
        mle: mle_coeffs_vec,
        commitment: commitment_values,
        blinding_factors_matrix,
    }
}

/// The prover's view of the commitment to the input layer, which includes the
/// blinding factors and the plaintext values.
#[derive(Clone)]
pub struct HyraxProverInputCommitment<C: PrimeOrderCurve> {
    /// The plaintext values
    pub mle: MultilinearExtension<C::Scalar>,
    /// The verifier's view of the commitment
    pub commitment: Vec<C>,
    /// The blinding factors used in the commitment
    pub blinding_factors_matrix: Vec<C::Scalar>,
}

/// Verifies a claim by evaluating the MLE at the challenge point and checking
/// that the result.
pub fn verify_claim<F: Field>(mle_vec: &[F], claim: &RawClaim<F>) {
    let mut mle = DenseMle::new_from_raw(mle_vec.to_vec(), LayerId::Input(0));
    mle.index_mle_indices(0);

    let eval = if mle.num_free_vars() != 0 {
        let mut eval = None;
        for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
            eval = mle.fix_variable(curr_bit, chal);
        }
        debug_assert_eq!(mle.len(), 1);
        eval.unwrap()
    } else {
        RawClaim::new(vec![], mle.mle.value())
    };

    assert_eq!(eval.get_point(), claim.get_point());
    assert_eq!(eval.get_eval(), claim.get_eval());
}

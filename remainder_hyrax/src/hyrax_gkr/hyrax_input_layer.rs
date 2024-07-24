use std::marker::PhantomData;

use itertools::Itertools;
use rand::Rng;
use remainder::{
    input_layer::{public_input_layer::PublicInputLayer, random_input_layer::RandomInputLayer},
    layer::LayerId,
};
use remainder_shared_types::{curves::PrimeOrderCurve, transcript::Transcript};

use crate::{
    hyrax_pcs::HyraxPCSProof,
    hyrax_primitives::{
        proof_of_claim_agg::ProofOfClaimAggregation, proof_of_equality::ProofOfEquality,
    },
    pedersen::{CommittedScalar, PedersenCommitter},
    utils::vandermonde::VandermondeInverse,
};

use super::hyrax_layer::HyraxClaim;

/// FIXME revise doc
/// FIXME: temporary fix to work with hyrax input layer proofs and the generic input layer proof for
/// [InputLayerEnum]. Need this for circuits that use multiple different types of input layers.
pub enum InputProofEnum<C: PrimeOrderCurve, Tr: Transcript<C::Scalar, C::Base>> {
    HyraxInputLayerProof(HyraxInputLayerProof<C, Tr>),
    PublicInputLayerProof(
        PublicInputLayer<C, C::Scalar, Tr>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ),
    RandomInputLayerProof(
        RandomInputLayer<C, C::Scalar, C::Base, Tr>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ),
}

/// The appropriate proof structure for a [HyraxInputLayer], which includes
/// the [ProofOfClaimAggregation], and the appropriate opening proof for opening
/// the polynomial commitment at a random evaluation point.

pub struct HyraxInputLayerProof<C: PrimeOrderCurve, Tr: Transcript<C::Scalar, C::Base>> {
    /// The ID of the layer that this is a proof for
    pub layer_id: LayerId,
    /// The proof of claim aggregation for this layer
    pub claim_agg_proof: ProofOfClaimAggregation<C>,
    /// The commitment to the input polynomial
    pub input_commitment: Vec<C>,
    /// The evaluation proof for the polynomial evaluated at a random point
    pub evaluation_proof: HyraxPCSProof<C>,
    /// The proof of equality that the aggregated claim point and the
    /// commitment in the evaluation proof are indeed to the same value
    pub proof_of_equality: ProofOfEquality<C>,
    _marker: PhantomData<Tr>,
}

impl<C: PrimeOrderCurve, Tr: Transcript<C::Scalar, C::Base>> HyraxInputLayerProof<C, Tr> {
    pub fn prove(
        input_layer: &HyraxInputLayer<C, Tr>,
        commitment: &Vec<C>,
        committed_claims: &Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
        converter: &mut VandermondeInverse<C::Scalar>,
    ) -> Self {
        // Calculate the coefficients of the polynomial that interpolates the claims
        // NB we don't use aggregate_claims here because the sampling of the evaluation
        // point for the aggregate claim needs to happen elsewhere in Hyrax.
        let claims = ClaimGroup::new(
            committed_claims
                .iter()
                .map(|committed_claim| committed_claim.to_claim())
                .collect_vec(),
        )
        .unwrap();
        let wlx_evals = input_layer.compute_claim_wlx(&claims).unwrap();
        let interpolant_coeffs = converter.convert_to_coefficients(wlx_evals);

        let (proof_of_claim_agg, aggregated_claim): (
            ProofOfClaimAggregation<C>,
            HyraxClaim<C::Scalar, CommittedScalar<C>>,
        ) = ProofOfClaimAggregation::prove(
            &committed_claims,
            &interpolant_coeffs,
            &committer,
            blinding_rng,
            transcript,
        );

        let evaluation_proof = HyraxPCSProof::prove(
            input_layer.log_num_cols,
            &input_layer.mle,
            &aggregated_claim.point,
            &aggregated_claim.evaluation.value,
            committer,
            input_layer.blinding_factor_eval,
            blinding_rng,
            transcript,
            &input_layer.blinding_factors_matrix,
        );

        let proof_of_equality = ProofOfEquality::prove(
            &aggregated_claim.evaluation,
            &evaluation_proof.commitment_to_evaluation,
            &committer,
            blinding_rng,
            transcript,
        );

        HyraxInputLayerProof {
            layer_id: input_layer.layer_id,
            input_commitment: commitment.clone(),
            claim_agg_proof: proof_of_claim_agg,
            evaluation_proof,
            proof_of_equality,
            _marker: PhantomData,
        }
    }

    /// Verify a Hyrax Input Layer proof by verifying the inner proof of claim aggregation,
    /// and then verifying the opening proof by checking the claim.
    pub fn verify(
        &self,
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
    ) {
        // Verify the proof of claim aggregation
        let agg_claim = self.claim_agg_proof.verify(
            &claim_commitments,
            &self.evaluation_proof.aux.committer,
            transcript,
        );

        // Verify the actual "evaluation" polynomial committed to at the random point
        self.evaluation_proof.verify_hyrax_evaluation_proof(
            self.evaluation_proof.aux.log_n_cols,
            &self.evaluation_proof.aux.committer,
            &self.input_commitment,
            &agg_claim.point,
            transcript,
        );

        // Proof of equality for the aggregated claim and the evaluation proof commitment
        self.proof_of_equality.verify(
            agg_claim.evaluation,
            self.evaluation_proof.commitment_to_evaluation.commitment,
            &committer,
            transcript,
        );
    }
}

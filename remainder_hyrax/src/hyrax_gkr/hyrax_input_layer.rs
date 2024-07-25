use ark_std::log2;
use itertools::Itertools;
use rand::{rngs::OsRng, Rng};
use rand_chacha::ChaCha20Rng;
use remainder::{
    claims::wlx_eval::claim_group::ClaimGroup,
    input_layer::{public_input_layer::PublicInputLayer, random_input_layer::RandomInputLayer},
    layer::LayerId,
    mle::dense::DenseMle,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve, transcript::ec_transcript::ECProverTranscript,
};

use crate::{
    hyrax_pcs::{HyraxPCSProof, MleCoefficientsVector},
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
pub enum InputProofEnum<C: PrimeOrderCurve> {
    HyraxInputLayerProof(HyraxInputLayerProof<C>),
    PublicInputLayerProof(
        PublicInputLayer<C::Scalar>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ),
    RandomInputLayerProof(
        RandomInputLayer<C::Scalar>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ),
}

/// The appropriate proof structure for a [HyraxInputLayer], which includes
/// the [ProofOfClaimAggregation], and the appropriate opening proof for opening
/// the polynomial commitment at a random evaluation point.

pub struct HyraxInputLayerProof<C: PrimeOrderCurve> {
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
}

impl<C: PrimeOrderCurve> HyraxInputLayerProof<C> {
    pub fn prove(
        input_layer: &HyraxInputLayer<C>,
        commitment: &Vec<C>,
        committed_claims: &Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
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
        }
    }

    /// Verify a Hyrax Input Layer proof by verifying the inner proof of claim aggregation,
    /// and then verifying the opening proof by checking the claim.
    pub fn verify(
        &self,
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECProverTranscript<C>,
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

pub struct HyraxInputLayer<C: PrimeOrderCurve> {
    pub mle: MleCoefficientsVector<C>,
    pub log_num_cols: usize,
    // NB committer wouldn't belong here, except that the committers need
    // to be available to for the implementation of the commit() method, whose signature is defined
    // by a trait.
    pub committer: PedersenCommitter<C>,
    pub blinding_factors_matrix: Vec<C::Scalar>,
    pub blinding_factor_eval: C::Scalar,
    pub(crate) layer_id: LayerId,
    comm: Option<Vec<C>>,
}

impl<C: PrimeOrderCurve> HyraxInputLayer<C> {
    pub fn new_with_committer(
        mle: DenseMle<C::Scalar>,
        layer_id: LayerId,
        committer: PedersenCommitter<C>,
    ) -> Self {
        let mle_len = mle.mle_ref().bookkeeping_table.len();
        assert!(mle_len.is_power_of_two());

        let mle_coefficients_vector = MleCoefficientsVector::ScalarFieldVector(mle.mle);
        let log_num_cols = (log2(mle_len) / 2) as usize;
        let num_rows = mle_len / (1 << log_num_cols);

        let mut seed_matrix = [0u8; 32];
        OsRng.fill_bytes(&mut seed_matrix);
        let mut prng = ChaCha20Rng::from_seed(seed_matrix);

        let blinding_factors_matrix = (0..num_rows)
            .map(|_| C::Scalar::random(&mut prng))
            .collect_vec();
        let mut seed_eval = [0u8; 32];
        OsRng.fill_bytes(&mut seed_eval);
        let mut prng = ChaCha20Rng::from_seed(seed_matrix);

        let blinding_factor_eval = C::Scalar::random(&mut prng);

        Self {
            mle: mle_coefficients_vector,
            layer_id,
            log_num_cols,
            committer,
            blinding_factors_matrix,
            blinding_factor_eval,
            comm: None,
        }
    }

    fn new_with_mle_coeff_vec(
        mle_coefficients_vector: MleCoefficientsVector<C>,
        layer_id: LayerId,
    ) -> Self {
        assert!(mle_coefficients_vector.len().is_power_of_two());
        let log_num_cols = (log2(mle_coefficients_vector.len()) / 2) as usize;
        let num_rows = mle_coefficients_vector.len() / (1 << log_num_cols);
        let committer = PedersenCommitter::new(
            1 << log_num_cols + 1,
            "abcdefghijklmnopqrstuvwxyz qwertyuiop",
            None,
        );

        let mut seed_matrix = [0u8; 32];
        OsRng.fill_bytes(&mut seed_matrix);
        let mut prng = ChaCha20Rng::from_seed(seed_matrix);
        let blinding_factors_matrix = (0..num_rows)
            .map(|_| C::Scalar::random(&mut prng))
            .collect_vec();
        let mut seed_eval = [0u8; 32];
        OsRng.fill_bytes(&mut seed_eval);
        let mut prng = ChaCha20Rng::from_seed(seed_eval);
        let blinding_factor_eval = C::Scalar::random(&mut prng);

        Self {
            mle: mle_coefficients_vector,
            layer_id,
            log_num_cols,
            committer,
            blinding_factors_matrix,
            blinding_factor_eval,
            comm: None,
        }
    }

    /// Creates new Ligero input layer WITH a precomputed Ligero commitment
    pub fn new_with_hyrax_commitment(
        mle: MleCoefficientsVector<C>,
        layer_id: LayerId,
        committer: PedersenCommitter<C>,
        blinding_factors_matrix: Vec<C::Scalar>,
        log_num_cols: usize,
        commitment: Vec<C>,
    ) -> Self {
        let mut seed_eval = [0u8; 32];
        OsRng.fill_bytes(&mut seed_eval);
        OsRng.fill_bytes(&mut seed_eval);
        let mut prng = ChaCha20Rng::from_seed(seed_eval);
        let blinding_factor_eval = C::Scalar::random(&mut prng);

        Self {
            mle,
            layer_id,
            log_num_cols,
            committer,
            blinding_factors_matrix,
            blinding_factor_eval,
            comm: Some(commitment),
        }
    }
}

use ark_std::{cfg_into_iter, log2};
use ff::Field;
use itertools::Itertools;
use rand::{rngs::OsRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use remainder::{
    claims::wlx_eval::{claim_group::ClaimGroup, get_num_wlx_evaluations},
    input_layer::{
        // hyrax_placeholder_input_layer::HyraxPlaceholderInputLayer,
        // hyrax_precommit_placeholder_input_layer::HyraxPrecommitPlaceholderInputLayer,
        hyrax_placeholder_input_layer::HyraxPlaceholderInputLayer,
        hyrax_precommit_placeholder_input_layer::HyraxPrecommitPlaceholderInputLayer,
        public_input_layer::PublicInputLayer,
        random_input_layer::RandomInputLayer,
        InputLayer,
    },
    layer::LayerId,
    mle::{evals::MultilinearExtension, Mle},
    sumcheck::evaluate_at_a_point,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve,
    transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript},
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

/// FIXME: temporary fix to work with hyrax input layer proofs and the generic input layer proof for
/// [HyraxCircuitInputLayerEnum]. Need this for circuits that use multiple different types of input layers.
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

/// FIXME: temporary fix to work with hyrax input layers and the generic input layers for
/// [InputLayerEnum] in `remainder_prover`. Need this for circuits that use multiple different
/// types of input layers.
pub enum HyraxCircuitInputLayerEnum<C: PrimeOrderCurve> {
    HyraxInputLayer(HyraxInputLayer<C>),
    PublicInputLayer(PublicInputLayer<C::Scalar>),
    RandomInputLayer(RandomInputLayer<C::Scalar>),
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
        let wlx_evals = input_layer.compute_claim_wlx(&claims);
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
        transcript: &mut impl ECVerifierTranscript<C>,
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
    pub fn new_from_placeholder_with_committer(
        hyrax_placeholder_il: HyraxPlaceholderInputLayer<C::Scalar>,
        committer: PedersenCommitter<C>,
    ) -> Self {
        HyraxInputLayer::new_with_committer(
            hyrax_placeholder_il.mle.clone(),
            hyrax_placeholder_il.layer_id().clone(),
            committer,
        )
    }

    pub fn new_with_committer(
        mle: MultilinearExtension<C::Scalar>,
        layer_id: LayerId,
        committer: PedersenCommitter<C>,
    ) -> Self {
        let mle_len = mle.f.len();
        assert!(mle_len.is_power_of_two());

        let mle_coefficients_vector = MleCoefficientsVector::ScalarFieldVector(mle.f.to_vec());
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

    pub fn new_from_placeholder_with_commitment(
        hyrax_placeholder_il: HyraxPrecommitPlaceholderInputLayer<C::Scalar>,
        committer: PedersenCommitter<C>,
        blinding_factors_matrix: Vec<C::Scalar>,
        log_num_cols: usize,
        commitment: Vec<C>,
    ) -> Self {
        HyraxInputLayer::new_with_hyrax_commitment(
            MleCoefficientsVector::ScalarFieldVector(hyrax_placeholder_il.mle.f.to_vec()),
            hyrax_placeholder_il.layer_id().clone(),
            committer,
            blinding_factors_matrix,
            log_num_cols,
            commitment,
        )
    }

    /// Creates new Hyrax input layer WITH a precomputed Hyrax commitment
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

    /// Computes the V_d(l(x)) evaluations for this input layer V_d for claim aggregation.
    fn compute_claim_wlx(&self, claims: &ClaimGroup<C::Scalar>) -> Vec<C::Scalar> {
        let mle_coeffs = self.mle.convert_to_scalar_field();
        let mle = MultilinearExtension::new(mle_coeffs);
        let num_claims = claims.get_num_claims();
        let claim_vecs = claims.get_claim_points_matrix();
        let claimed_vals = claims.get_results();
        let num_idx = claims.get_num_vars();

        // get the number of evaluations
        let (num_evals, _common_idx, _) = get_num_wlx_evaluations(claim_vecs);
        let chal_point = &claim_vecs[0];

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<C::Scalar> = cfg_into_iter!(num_claims..num_evals)
            // let next_evals: Vec<F> = (num_claims..num_evals).into_iter()
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<C::Scalar> = cfg_into_iter!(0..num_idx)
                    // let new_chal: Vec<F> = (0..num_idx).into_iter()
                    .map(|claim_idx| {
                        let evals: Vec<C::Scalar> = cfg_into_iter!(&claim_vecs)
                            // let evals: Vec<F> = (&claim_vecs).into_iter()
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, C::Scalar::from(idx as u64)).unwrap()
                    })
                    .collect();

                let mut fix_mle = mle.clone();
                {
                    new_chal
                        .into_iter()
                        .for_each(|chal| fix_mle.fix_variable(chal));
                    assert_eq!(fix_mle.f.len(), 1);
                    fix_mle.f[0]
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        wlx_evals
    }
}

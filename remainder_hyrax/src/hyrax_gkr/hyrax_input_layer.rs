use ark_std::cfg_into_iter;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    claims::{
        wlx_eval::{claim_group::ClaimGroup, get_num_wlx_evaluations},
        Claim,
    },
    input_layer::InputLayerDescription,
    layer::{regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION, LayerId},
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    sumcheck::evaluate_at_a_point,
};
use remainder_shared_types::{
    curves::PrimeOrderCurve, transcript::ec_transcript::ECTranscriptTrait
};
use remainder_shared_types::{ff_field, Field};

use crate::{
    hyrax_pcs::{HyraxPCSEvaluationProof, MleCoefficientsVector},
    hyrax_primitives::{
        proof_of_claim_agg::ProofOfClaimAggregation, proof_of_equality::ProofOfEquality,
    },
    pedersen::{CommittedScalar, PedersenCommitter},
    utils::vandermonde::VandermondeInverse,
};

use super::hyrax_layer::HyraxClaim;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// The proof structure for a [HyraxInputLayer], which includes the [ProofOfClaimAggregation], and
/// the appropriate opening proof for opening the polynomial commitment at a random evaluation
/// point.
pub struct HyraxInputLayerProof<C: PrimeOrderCurve> {
    /// The ID of the layer that this is a proof for
    pub layer_id: LayerId,
    /// The proof of claim aggregation for this layer
    pub claim_agg_proof: ProofOfClaimAggregation<C>,
    /// The commitment to the input polynomial
    pub input_commitment: Vec<C>,
    /// The evaluation proof for the polynomial evaluated at a random point
    pub evaluation_proof: HyraxPCSEvaluationProof<C>,
    /// The proof of equality that the aggregated claim point and the
    /// commitment in the evaluation proof are indeed to the same value
    pub proof_of_equality: ProofOfEquality<C>,
}

impl<C: PrimeOrderCurve> HyraxInputLayerProof<C> {
    pub fn prove(
        input_layer_desc: &HyraxInputLayerDescription,
        commitment: &HyraxInputCommitment<C>,
        committed_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        transcript: &mut impl ECTranscriptTrait<C>,
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
        let wlx_evals = compute_claim_wlx(&commitment.mle.convert_to_scalar_field(), &claims);
        let interpolant_coeffs = converter.convert_to_coefficients(wlx_evals);

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

        let evaluation_proof = HyraxPCSEvaluationProof::prove(
            input_layer_desc.log_num_cols,
            &commitment.mle,
            &aggregated_claim.point,
            &aggregated_claim.evaluation.value,
            committer,
            blinding_rng,
            transcript,
            &commitment.blinding_factors_matrix,
        );

        let proof_of_equality = ProofOfEquality::prove(
            &aggregated_claim.evaluation,
            &evaluation_proof.commitment_to_evaluation,
            committer,
            blinding_rng,
            transcript,
        );

        HyraxInputLayerProof {
            layer_id: input_layer_desc.layer_id,
            input_commitment: commitment.commitment.clone(),
            claim_agg_proof: proof_of_claim_agg,
            evaluation_proof,
            proof_of_equality,
        }
    }

    /// Verify a Hyrax Input Layer proof by verifying the inner proof of claim aggregation,
    /// and then verifying the opening proof by checking the claim.
    pub fn verify(
        &self,
        input_layer_desc: &HyraxInputLayerDescription,
        claim_commitments: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // Verify the proof of claim aggregation
        let agg_claim = self.claim_agg_proof.verify(
            claim_commitments,
            committer,
            transcript,
        );

        // Verify the actual "evaluation" polynomial committed to at the random point
        self.evaluation_proof.verify(
            input_layer_desc.log_num_cols,
            committer,
            &self.input_commitment,
            &agg_claim.point,
            transcript,
        );

        // Proof of equality for the aggregated claim and the evaluation proof commitment
        self.proof_of_equality.verify(
            agg_claim.evaluation,
            self.evaluation_proof.commitment_to_evaluation.commitment,
            committer,
            transcript,
        );
    }
}

#[derive(Clone, Debug, PartialEq)]
/// The circuit description of a [HyraxInputLayer]. Stores the shape information of this layer.
/// All of the functionality of Hyrax input layers are taken care of in `remainder_hyrax/`, so
/// this is meant just to generate a circuit description.
pub struct HyraxInputLayerDescription {
    /// The input layer ID.
    pub layer_id: LayerId,
    /// The number of variables this Hyrax Input Layer is on.
    pub num_bits: usize,
    /// The log number of columns in the matrix form of the data that
    /// will be committed to in this input layer.
    pub log_num_cols: usize,
}

impl HyraxInputLayerDescription {
    /// Create a [HyraxInputLayerDescription] specifying the use of a square matrix ("default
    /// setup"; build the struct directly for custom setup).
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
    /// Convert an [InputLayerDescription] into a [HyraxInputLayerDescription] with a square matrix.
    fn from(input_layer_desc: InputLayerDescription) -> Self {
        HyraxInputLayerDescription::new(input_layer_desc.layer_id, input_layer_desc.num_vars)
    }
}

/// Given a [HyraxInputLayerDescription] and values for its MLE, compute the [HyraxInputCommitment]
/// for the input layer.
pub fn commit_to_input_values<C: PrimeOrderCurve>(
    input_layer_desc: &HyraxInputLayerDescription,
    input_mle: &MultilinearExtension<C::Scalar>, 
    committer: &PedersenCommitter<C>,
    mut rng: &mut impl Rng,
) -> HyraxInputCommitment<C> {
    let num_rows = 1 << (input_layer_desc.num_bits - input_layer_desc.log_num_cols);
    // Sample the blinding factors
    let mut blinding_factors_matrix = vec![C::Scalar::ZERO; num_rows];
    for i in 0..num_rows {
        blinding_factors_matrix[i] = C::Scalar::random(&mut rng);
    }
    let mle_coeffs_vec = MleCoefficientsVector::ScalarFieldVector(input_mle.get_evals_vector().clone()); 
    let commitment_values = HyraxPCSEvaluationProof::compute_matrix_commitments(
        input_layer_desc.log_num_cols,
        &mle_coeffs_vec,
        committer,
        &blinding_factors_matrix,
    );
    HyraxInputCommitment {
        mle: mle_coeffs_vec,
        commitment: commitment_values,
        blinding_factors_matrix,
    }
}

/// The prover's view of the commitment to the input layer, which includes the blinding factors and the plaintext values.
#[derive(Clone)]
pub struct HyraxInputCommitment<C: PrimeOrderCurve> {
    /// The plaintext values
    pub mle: MleCoefficientsVector<C>,
    /// The verifier's view of the commitment
    pub commitment: Vec<C>,
    /// The blinding factors used in the commitment
    pub blinding_factors_matrix: Vec<C::Scalar>,
}

/// Computes the V_d(l(x)) evaluations for this input layer V_d for claim aggregation.
fn compute_claim_wlx<F: Field>(mle_vec: &[F], claims: &ClaimGroup<F>) -> Vec<F> {
    let mle = MultilinearExtension::new(mle_vec.to_owned());
    let num_claims = claims.get_num_claims();
    let claim_vecs = claims.get_claim_points_matrix();
    let claimed_vals = claims.get_results();
    let num_idx = claims.get_num_vars();

    // get the number of evaluations
    let num_evals = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION {
        let (num_evals, _, _) = get_num_wlx_evaluations(claim_vecs);
        num_evals
    } else {
        ((num_claims - 1) * num_idx) + 1
    };

    // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
    let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
        // let next_evals: Vec<F> = (num_claims..num_evals).into_iter()
        .map(|idx| {
            // get the challenge l(idx)
            let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                // let new_chal: Vec<F> = (0..num_idx).into_iter()
                .map(|claim_idx| {
                    let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                        // let evals: Vec<F> = (&claim_vecs).into_iter()
                        .map(|claim| claim[claim_idx])
                        .collect();
                    evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
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

/// Verifies a claim by evaluating the MLE at the challenge point and checking that the result.
pub fn verify_claim<F: Field>(mle_vec: &[F], claim: &Claim<F>) {
    let mut mle = DenseMle::new_from_raw(mle_vec.to_vec(), LayerId::Input(0));
    mle.index_mle_indices(0);

    let eval = if mle.num_free_vars() != 0 {
        let mut eval = None;
        for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
            eval = mle.fix_variable(curr_bit, chal);
        }
        debug_assert_eq!(mle.bookkeeping_table().len(), 1);
        eval.unwrap()
    } else {
        Claim::new(vec![], mle.mle[0])
    };

    assert_eq!(eval.get_point(), claim.get_point());
    assert_eq!(eval.get_result(), claim.get_result());
}

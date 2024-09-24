use ark_std::{cfg_into_iter, log2};
use itertools::Itertools;
use rand::{rngs::OsRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use remainder::{
    claims::{
        wlx_eval::{claim_group::ClaimGroup, get_num_wlx_evaluations},
        Claim,
    },
    input_layer::{
        enum_input_layer::{CircuitInputLayerEnum, InputLayerEnum},
        public_input_layer::PublicInputLayer,
        CircuitInputLayer, InputLayer,
    },
    layer::{regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION, LayerId},
    layouter::nodes::circuit_inputs::HyraxInputDType,
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    sumcheck::evaluate_at_a_point,
};
use remainder_shared_types::ff_field;
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
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// FIXME: temporary fix to work with hyrax input layer proofs and the generic input layer proof for
/// [HyraxCircuitInputLayerEnum]. Need this for circuits that use multiple different types of input layers.
pub enum InputProofEnum<C: PrimeOrderCurve> {
    HyraxInputLayerProof(HyraxInputLayerProof<C>),
    PublicInputLayerProof(
        PublicInputLayer<C::Scalar>,
        Vec<HyraxClaim<C::Scalar, CommittedScalar<C>>>,
    ),
}
/// FIXME: temporary fix to work with hyrax input layers and the generic input layers for
/// [InputLayerEnum] in `remainder_prover`. Need this for circuits that use multiple different
/// types of input layers.
pub enum HyraxInputLayerEnum<C: PrimeOrderCurve> {
    HyraxInputLayer(HyraxInputLayer<C>),
    PublicInputLayer(PublicInputLayer<C::Scalar>),
}

impl<C: PrimeOrderCurve> HyraxInputLayerEnum<C> {
    pub fn from_circuit_input_layer_enum(
        circuit_input_layer_enum: CircuitInputLayerEnum<C::Scalar>,
        input_layer_mle: MultilinearExtension<C::Scalar>,
        precommit: Option<HyraxProverCommitmentEnum<C>>,
    ) -> Self {
        match circuit_input_layer_enum {
            CircuitInputLayerEnum::LigeroInputLayer(_circuit_ligero_input_layer) => {
                panic!("hyrax does not support ligero pcs")
            }
            CircuitInputLayerEnum::PublicInputLayer(circuit_public_input_layer) => {
                assert!(precommit.is_none());
                let input_layer_enum = circuit_public_input_layer
                    .convert_into_prover_input_layer(input_layer_mle, &None);
                Self::from_input_layer_enum(input_layer_enum)
            }
            CircuitInputLayerEnum::HyraxInputLayer(_circuit_hyrax_input_layer) => {
                unimplemented!("We handle the hyrax case separately")
            }
        }
    }
    pub fn from_input_layer_enum(input_layer_enum: InputLayerEnum<C::Scalar>) -> Self {
        match input_layer_enum {
            InputLayerEnum::LigeroInputLayer(_ligero_input_layer) => {
                panic!("hyrax does not support ligero pcs");
            }
            InputLayerEnum::PublicInputLayer(public_input_layer) => {
                HyraxInputLayerEnum::PublicInputLayer(*public_input_layer)
            }
        }
    }
    pub fn layer_id(&self) -> LayerId {
        match self {
            HyraxInputLayerEnum::HyraxInputLayer(hyrax_layer) => hyrax_layer.layer_id,
            HyraxInputLayerEnum::PublicInputLayer(public_layer) => public_layer.layer_id(),
        }
    }
}

/// An Enum representing the types of commitments for each layer,
/// but from the prover's view. These are the precommit types
/// included in [HyraxInputLayerData].
#[derive(Debug, Clone)]
pub enum HyraxProverCommitmentEnum<C: PrimeOrderCurve> {
    HyraxCommitment((Vec<C>, Vec<C::Scalar>)),
    PublicCommitment(Vec<C::Scalar>),
}

/// An Enum representing the types of commitments for each layer,
/// but from the verifier's view. These are the commitment types
/// added to transcript.
#[derive(Debug, Clone)]
pub enum HyraxVerifierCommitmentEnum<C: PrimeOrderCurve> {
    HyraxCommitment(Vec<C>),
    PublicCommitment(Vec<C::Scalar>),
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
        commitment: &[C],
        committed_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
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
            committed_claims,
            &interpolant_coeffs,
            committer,
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
            committer,
            blinding_rng,
            transcript,
        );

        HyraxInputLayerProof {
            layer_id: input_layer.layer_id,
            input_commitment: commitment.to_vec(),
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
            claim_commitments,
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
            committer,
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
    pub comm: Option<Vec<C>>,
}

impl<C: PrimeOrderCurve> HyraxInputLayer<C> {
    /// Just a wrapper around the corresponding [HyraxPCSProof] function.
    pub fn commit(&mut self) -> Vec<C> {
        assert!(
            self.comm.is_none(),
            "should not be committing if there is already a precommit!"
        );
        let comm = HyraxPCSProof::compute_matrix_commitments(
            self.log_num_cols,
            &self.mle,
            &self.committer,
            &self.blinding_factors_matrix,
        );
        self.comm = Some(comm.clone());
        comm
    }

    fn to_mle_coeffs_vec(
        mle: MultilinearExtension<C::Scalar>,
        maybe_hyrax_input_dtype: &Option<HyraxInputDType>,
    ) -> MleCoefficientsVector<C> {
        if let Some(hyrax_input_dtype) = maybe_hyrax_input_dtype {
            MleCoefficientsVector::convert_from_scalar_field(
                mle.get_evals_vector(),
                hyrax_input_dtype,
            )
        } else {
            MleCoefficientsVector::ScalarFieldVector(mle.f.to_vec())
        }
    }

    pub fn new_with_committer(
        mle: MultilinearExtension<C::Scalar>,
        layer_id: LayerId,
        committer: &PedersenCommitter<C>,
        maybe_hyrax_input_dtype: &Option<HyraxInputDType>,
    ) -> Self {
        let mle_len = mle.f.len();
        assert!(mle_len.is_power_of_two());

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
            mle: Self::to_mle_coeffs_vec(mle, maybe_hyrax_input_dtype),
            layer_id,
            log_num_cols,
            committer: committer.clone(),
            blinding_factors_matrix,
            blinding_factor_eval,
            comm: None,
        }
    }

    /// Creates new Hyrax input layer WITH a precomputed Hyrax commitment
    pub fn new_with_hyrax_commitment(
        mle: MultilinearExtension<C::Scalar>,
        maybe_hyrax_input_dtype: &Option<HyraxInputDType>,
        layer_id: LayerId,
        committer: &PedersenCommitter<C>,
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
            mle: Self::to_mle_coeffs_vec(mle, maybe_hyrax_input_dtype),
            layer_id,
            log_num_cols,
            committer: committer.clone(),
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
        let num_evals = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION {
            let (num_evals, _, _) = get_num_wlx_evaluations(claim_vecs);
            num_evals
        } else {
            ((num_claims - 1) * num_idx) + 1
        };

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

pub fn verify_public_and_random_input_layer<C: PrimeOrderCurve>(
    mle_vec: &[C::Scalar],
    claim: &Claim<C::Scalar>,
) {
    let mut mle = DenseMle::new_from_raw(mle_vec.to_vec(), LayerId::Input(0));
    mle.index_mle_indices(0);

    let eval = if mle.num_iterated_vars() != 0 {
        let mut eval = None;
        for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
            eval = mle.fix_variable(curr_bit, chal);
        }
        debug_assert_eq!(mle.bookkeeping_table().len(), 1);
        eval.unwrap()
    } else {
        Claim::new(vec![], mle.current_mle[0])
    };

    assert_eq!(eval.get_point(), claim.get_point());
    assert_eq!(eval.get_result(), claim.get_result());
}

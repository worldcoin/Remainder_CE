use std::collections::HashMap;

use ark_std::{end_timer, start_timer};
use itertools::Itertools;
use rand::Rng;
use remainder::{
    claims::RawClaim,
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

use crate::hyrax_pcs::HyraxPCSEvaluationProof;

use super::hyrax_layer::HyraxClaim;

/// The proof structure for a Hyrax input layer.
pub struct HyraxInputLayerProof<C: PrimeOrderCurve> {
    /// The ID of the layer that this is a proof for
    pub layer_id: LayerId,
    /// The commitment to the input polynomial
    pub input_commitment: Vec<C>,
    /// The evaluation points and evaluation proofs for the input layer claims, sorted by evaluation point.
    pub evaluation_proofs: Vec<(Vec<C::Scalar>, HyraxPCSEvaluationProof<C>)>,
}

impl<C: PrimeOrderCurve> HyraxInputLayerProof<C> {
    /// Create a proof of the claims on a Hyrax input layer.
    /// # Arguments:
    /// * `input_layer_desc` - The description of the input layer
    /// * `prover_commitment` - The commitment to the input layer
    /// * `committed_claims` - The claims made on this layer (in any order)
    pub fn prove(
        input_layer_desc: &HyraxInputLayerDescription,
        prover_commitment: &HyraxProverInputCommitment<C>,
        committed_claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        committer: &PedersenCommitter<C>,
        blinding_rng: &mut impl Rng,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Self {
        let eval_proof_timer = start_timer!(|| "eval proof timer");
        // Sort the claims by evaluation point
        let mut committed_claims = committed_claims.to_vec();
        committed_claims.sort_by(|a, b| a.point.cmp(&b.point));
        let evaluation_proofs = committed_claims
            .iter()
            .map(|claim| {
                let proof = HyraxPCSEvaluationProof::prove(
                    input_layer_desc.log_num_cols,
                    &prover_commitment.mle,
                    &claim.point,
                    &claim.evaluation,
                    committer,
                    blinding_rng,
                    transcript,
                    &prover_commitment.blinding_factors_matrix,
                );
                (claim.point.clone(), proof)
            })
            .collect_vec();
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

        // Check there are the same number of claims as evaluation proofs
        assert_eq!(self.evaluation_proofs.len(), claim_commitments.len());

        // Verify each evaluation proof
        claim_commitments
            .iter()
            .zip(&self.evaluation_proofs)
            .for_each(|(claim, (eval_point, eval_proof))| {
                assert_eq!(claim.point, *eval_point);
                assert_eq!(claim.evaluation, eval_proof.commitment_to_evaluation);
                eval_proof.verify(
                    input_layer_desc.log_num_cols,
                    committer,
                    &self.input_commitment,
                    &claim.point,
                    transcript,
                );
            });
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

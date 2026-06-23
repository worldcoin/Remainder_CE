use crate::gkr::ECTranscriptTrait;
use itertools::Itertools;
use rand::Rng;
use remainder::mle::{Mle, MleIndex};
use remainder::output_layer::{OutputLayer, OutputLayerDescription};
use serde::{Deserialize, Serialize};
use shared_types::curves::PrimeOrderCurve;
use shared_types::ff_field;
use shared_types::pedersen::{CommittedScalar, PedersenCommitter};

use crate::gkr::layer::HyraxClaim;

#[cfg(test)]
mod tests;

/// The proof structure for the proof of a Hyrax output layer, which
/// doesn't need anything other than whether the challenges the
/// output layer was evaluated on, so that the verifier can check
/// whether these match the transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxOutputLayerProof<C: PrimeOrderCurve> {
    /// The commitment to the claim that the output layer is making
    pub claim_commitment: C,
    /// The blinding factor of `claim_commitment`. It is revealed so that the
    /// verifier can check that a zero-test output layer's claim opens to zero,
    /// mirroring how other verifier-checked claims (e.g. claims on public
    /// inputs) are sent with their openings. The committed value is public
    /// (zero) for zero-test layers, so revealing the blinding leaks nothing.
    pub claim_blinding: C::Scalar,
}

impl<C: PrimeOrderCurve> HyraxOutputLayerProof<C> {
    /// Returns a HyraxOutputLayerProof and the claim that the output layer is making.
    pub fn prove(
        output_layer: &mut OutputLayer<C::Scalar>,
        transcript: &mut impl ECTranscriptTrait<C>,
        blinding_rng: &mut impl Rng,
        scalar_committer: &PedersenCommitter<C>,
    ) -> (Self, HyraxClaim<C::Scalar, CommittedScalar<C>>) {
        // Fix variable on the output layer in order to generate the claim on the previous layer
        let bindings: Vec<C::Scalar> = (0..output_layer.get_mle().num_free_vars())
            .map(|_idx| transcript.get_scalar_field_challenge("Challenge for claim on output"))
            .collect_vec();
        output_layer.fix_layer(&bindings).unwrap();
        let claim = output_layer.get_claim().unwrap();

        // Convert to a CommittedScalar claim.
        let blinding_factor = C::Scalar::random(blinding_rng);
        let claim_commit = scalar_committer.committed_scalar(&claim.get_eval(), &blinding_factor);
        let committed_claim = HyraxClaim {
            point: claim.get_point().to_vec(),
            to_layer_id: claim.get_to_layer_id(),
            evaluation: claim_commit,
        };
        let commitment = committed_claim.to_claim_commitment().evaluation;
        // Add the commitment to the transcript
        transcript.append_ec_point("Commitment to claim on output layer", commitment);

        (
            Self {
                claim_commitment: commitment,
                claim_blinding: blinding_factor,
            },
            committed_claim,
        )
    }

    /// This verify method does not do much: it takes the commitment to the evaluation provided by
    /// the prover, adds it to the transcript, and then returns a [HyraxClaim] that contains the
    /// challenges that it ITSELF draws from the transcript.
    ///
    /// Note that claims are generated from the `layer_desc`, not the `proof`!
    /// This ensures that a prover cannot cheat by creating a valid proof whose
    /// "shape" does not match that of the circuit description.
    pub fn verify(
        proof: &HyraxOutputLayerProof<C>,
        layer_desc: &OutputLayerDescription<C::Scalar>,
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> HyraxClaim<C::Scalar, C> {
        // Get the first set of challenges needed for the output layer.
        let bindings = layer_desc
            .mle
            .var_indices()
            .iter()
            .map(|idx| match idx {
                MleIndex::Fixed(bit) => C::Scalar::from(*bit as u64),
                MleIndex::Indexed(_) => {
                    transcript.get_scalar_field_challenge("Challenge for claim on output")
                }
                MleIndex::Free => panic!("MLEs should be indexed by this point"),
                _ => panic!("Unexpected MleIndex"),
            })
            .collect_vec();
        transcript.append_ec_point(
            "Commitment to claim on output layer",
            proof.claim_commitment,
        );

        // Soundness check for zero-test output layers: the committed claim must
        // open to zero. We recompute the commitment to zero using the revealed
        // blinding factor and require it to match. Without this check a
        // malicious prover could commit to the true non-zero output and still
        // produce an internally consistent proof, breaking soundness for any
        // zero-test circuit. This mirrors the plaintext verifier's
        // NonZeroEvalForZeroMle check in `remainder::output_layer`.
        assert!(
            !layer_desc.is_zero()
                || proof.claim_commitment
                    == committer.scalar_commit(&C::Scalar::ZERO, &proof.claim_blinding),
            "Zero-test output layer claim does not open to zero"
        );

        HyraxClaim {
            point: bindings,
            to_layer_id: layer_desc.mle.layer_id(),
            evaluation: proof.claim_commitment,
        }
    }
}

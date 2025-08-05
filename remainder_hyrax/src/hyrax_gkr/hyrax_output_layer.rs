use crate::hyrax_gkr::ECTranscriptTrait;
use itertools::Itertools;
use rand::Rng;
use remainder::mle::{Mle, MleIndex};
use remainder::output_layer::{OutputLayer, OutputLayerDescription};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use serde::{Deserialize, Serialize};

use super::hyrax_layer::HyraxClaim;

/// The proof structure for the proof of a Hyrax output layer, which
/// doesn't need anything other than whether the challenges the
/// output layer was evaluated on, so that the verifier can check
/// whether these match the transcript.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct HyraxOutputLayerProof<C: PrimeOrderCurve> {
    /// The commitment to the claim that the output layer is making
    pub claim_commitment: C,
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
        // Convert to a CommittedScalar claim
        let blinding_factor = &C::Scalar::random(blinding_rng);
        let claim_commit = scalar_committer.committed_scalar(&claim.get_eval(), blinding_factor);
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

        HyraxClaim {
            point: bindings,
            to_layer_id: layer_desc.mle.layer_id(),
            evaluation: proof.claim_commitment,
        }
    }
}

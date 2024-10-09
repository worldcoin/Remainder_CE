use itertools::Itertools;
use rand::Rng;
use remainder::layer::{Layer, LayerId};
use remainder::mle::{Mle, MleIndex};
use remainder::output_layer::{OutputLayer, OutputLayerDescription};
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::transcript::ec_transcript::ECTranscriptTrait;

use crate::pedersen::{CommittedScalar, PedersenCommitter};

use super::hyrax_layer::HyraxClaim;

/// The proof structure for the proof of a [HyraxOutputLayer], which
/// doesn't need anything other than whether the challenges the
/// output layer was evaluated on, so that the verifier can check
/// whether these match the transcript.
pub struct HyraxOutputLayerProof<C: PrimeOrderCurve> {
    /// The layer id
    pub layer_id: LayerId,
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
            .map(|_idx| transcript.get_scalar_field_challenge("output claim point"))
            .collect_vec();
        output_layer.fix_layer(&bindings).unwrap();
        let claim = output_layer.get_claim().unwrap();
        // Convert to a CommittedScalar claim
        // FIXME(Ben) write a helper for this.
        let blinding_factor = &C::Scalar::random(blinding_rng);
        let claim_commit = scalar_committer
            .committed_scalar(&claim.get_claim().get_result(), blinding_factor);
        let committed_claim = HyraxClaim {
            point: claim.get_claim().get_point().clone(),
            to_layer_id: claim.get_to_layer_id().unwrap(),
            evaluation: claim_commit,
        };
        let commitment = committed_claim.to_claim_commitment().evaluation;
        // Add the commitment to the transcript
        transcript.append_ec_point("output layer commit", commitment);

        (
            Self {
                layer_id: output_layer.get_mle().layer_id(),
                claim_commitment: commitment,
            },
            committed_claim,
        )
    }

    /// This verify method does not do much: it takes the commitment to the evaluation provided by
    /// the prover, adds it to the transcript, and then returns a [HyraxClaim] that contains the
    /// challenges that it ITSELF draws from the transcript.
    pub fn verify(
        proof: &HyraxOutputLayerProof<C>,
        layer_desc: &OutputLayerDescription<C::Scalar>,
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> HyraxClaim<C::Scalar, C> {
        // Get the first set of challenges needed for the output layer.
        let bindings = layer_desc.mle.var_indices().iter().map(|idx| match idx {
            MleIndex::Fixed(bit) => C::Scalar::from(*bit as u64),
            MleIndex::Indexed(_) => transcript.get_scalar_field_challenge("output claim point"),
            MleIndex::Free => panic!("MLEs should be indexed by this point"),
            _ => panic!("Unexpected MleIndex"),
        }).collect_vec();
        transcript.append_ec_point("output layer commit", proof.claim_commitment);

        HyraxClaim {
            point: bindings,
            to_layer_id: layer_desc.mle.layer_id(),
            evaluation: proof.claim_commitment,
        }
    }
}

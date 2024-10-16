use std::marker::PhantomData;

use itertools::Itertools;
use rand::Rng;
use remainder::mle::mle_enum::MleEnum;
use remainder::mle::{Mle, MleIndex};
use remainder::output_layer::mle_output_layer::MleOutputLayerDescription;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::pedersen::{CommittedScalar, PedersenCommitter};
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};

use super::hyrax_layer::HyraxClaim;

/// This is a wrapper around the existing [MleEnum], but suited in order
/// to produce Zero Knowledge evaluations using Hyrax.
pub struct HyraxOutputLayer<C: PrimeOrderCurve> {
    /// This is the MLE that this is a wrapper over. The output layer is always just an MLE.
    pub underlying_mle: MleEnum<C::Scalar>,
    pub _marker: PhantomData<C>,
}

impl<C: PrimeOrderCurve> HyraxOutputLayer<C> {
    /// This function will evaluate the output layer at a random point, which is the challenge.
    /// It will get these challenges from the transcript.
    pub fn fix_variable_on_challenge(
        &mut self,
        prover_transcript: &mut impl ECProverTranscript<C>,
    ) {
        let challenge: Vec<C::Scalar> = (0..self.underlying_mle.num_free_vars())
            .map(|_idx| prover_transcript.get_scalar_field_challenge("output claim point"))
            .collect_vec();
        self.underlying_mle.index_mle_indices(0);
        challenge
            .into_iter()
            .enumerate()
            .for_each(|(idx, chal_point)| {
                self.underlying_mle.fix_variable(idx, chal_point);
            })
    }

    /// This function will traverse the MLE in order to see which point it has been bound to, and
    /// return the point and the value it evaluates to in the form of a [Claim].
    pub fn get_claim(
        &mut self,
        blinding_rng: &mut impl Rng,
        scalar_committer: &PedersenCommitter<C>,
    ) -> HyraxClaim<C::Scalar, CommittedScalar<C>> {
        assert_eq!(self.underlying_mle.len(), 1);

        let layer_id = self.underlying_mle.layer_id();
        let claim_chal = if !self.underlying_mle.mle_indices().is_empty() {
            self.underlying_mle
                .mle_indices()
                .iter()
                .map(|index| index.val().unwrap())
                .collect_vec()
        } else {
            vec![]
        };
        let blinding_factor = &C::Scalar::random(blinding_rng);
        let claim_commit =
            scalar_committer.committed_scalar(&self.underlying_mle.first(), blinding_factor);

        HyraxClaim {
            point: claim_chal,
            to_layer_id: layer_id,
            evaluation: claim_commit,
        }
    }
}

/// The proof structure for the proof of a [HyraxOutputLayer], which
/// doesn't need anything other than whether the challenges the
/// output layer was evaluated on, so that the verifier can check
/// whether these match the transcript.
pub struct HyraxOutputLayerProof<C: PrimeOrderCurve> {
    /// The commitment to the claim that the output layer is making
    pub claim_commitment: C,
}

impl<C: PrimeOrderCurve> HyraxOutputLayerProof<C> {
    /// Returns a HyraxOutputLayerProof and the claim that the output layer is making.
    pub fn prove(
        output_layer: &mut HyraxOutputLayer<C>,
        transcript: &mut impl ECProverTranscript<C>,
        blinding_rng: &mut impl Rng,
        scalar_committer: &PedersenCommitter<C>,
    ) -> (Self, HyraxClaim<C::Scalar, CommittedScalar<C>>) {
        // Fix variable on the output layer in order to generate the claim on the previous layer
        output_layer.fix_variable_on_challenge(transcript);
        let committed_claim = output_layer.get_claim(blinding_rng, scalar_committer);
        let commitment = committed_claim.to_claim_commitment().evaluation;
        // Add the commitment to the transcript
        transcript.append_ec_point("output layer commit", commitment);

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
    pub fn verify(
        proof: &HyraxOutputLayerProof<C>,
        layer_desc: &MleOutputLayerDescription<C::Scalar>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) -> HyraxClaim<C::Scalar, C> {
        // Get the first set of challenges needed for the output layer.

        let bindings = layer_desc
            .mle
            .mle_indices()
            .iter()
            .map(|mle_index| match mle_index {
                MleIndex::Fixed(val) => C::Scalar::from(*val as u64),
                MleIndex::Indexed(_) => transcript
                    .get_scalar_field_challenge("output claim point")
                    .unwrap(),

                _ => {
                    panic!("should not have bound or free variables here!")
                }
            })
            .collect_vec();

        let transcript_claim_commit = transcript.consume_ec_point("output layer commit").unwrap();
        assert_eq!(proof.claim_commitment, transcript_claim_commit);

        HyraxClaim {
            point: bindings,
            to_layer_id: layer_desc.mle.layer_id(),
            evaluation: proof.claim_commitment,
        }
    }
}

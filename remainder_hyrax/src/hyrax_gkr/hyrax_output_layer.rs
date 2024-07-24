use crate::{curves::PrimeOrderCurve, pedersen::CommittedScalar, pedersen::PedersenCommitter};
use crate::{
    layer::{product::HyraxClaim, LayerId},
    mle::{mle_enum::MleEnum, MleRef},
    utils::append_x_y_to_transcript_single,
};
use itertools::Itertools;
use rand::Rng;
use remainder_shared_types::halo2curves::group::ff::Field;
use remainder_shared_types::transcript::Transcript;
use serde::{Deserialize, Serialize};
use tracing_subscriber::Layer;

/// This is a wrapper around the existing [MleEnum], but suited in order
/// to produce Zero Knowledge evaluations using Hyrax.
#[derive(Clone)]
pub struct HyraxOutputLayer<C: PrimeOrderCurve> {
    /// This is the MLE that this is a wrapper over. The output layer is always just an MLE.
    pub underlying_mle: MleEnum<C::Scalar>,
}

/// Everything that is needed to describe an output layer, without any dependence on the actual input values that instantiate a circuit.
#[derive(Clone, Serialize, Deserialize)]
pub struct OutputLayerDescription {
    /// The number of variables in the output layer
    pub num_vars: usize,
    /// The layer id of the output layer
    pub layer_id: LayerId,
}

impl<C: PrimeOrderCurve> From<HyraxOutputLayer<C>> for OutputLayerDescription {
    /// Conversion from an output layer to its description
    fn from(output_layer: HyraxOutputLayer<C>) -> Self {
        Self {
            num_vars: output_layer.underlying_mle.original_mle_indices().len(),
            layer_id: output_layer.underlying_mle.get_layer_id(),
        }
    }
}

impl<C: PrimeOrderCurve> HyraxOutputLayer<C> {
    /// This function will evaluate the output layer at a random point, which is the challenge.
    /// It will get these challenges from the transcript.
    pub fn fix_variable_on_challenge(
        &mut self,
        prover_transcript: &mut impl Transcript<C::Scalar, C::Base>,
    ) {
        let challenge: Vec<C::Scalar> = (0..self.underlying_mle.num_vars())
            .map(|_idx| {
                prover_transcript
                    .get_scalar_field_challenge("output claim point")
                    .unwrap()
            })
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
        assert_eq!(self.underlying_mle.bookkeeping_table().len(), 1);

        let layer_id = self.underlying_mle.get_layer_id();
        let claim_chal = if self.underlying_mle.mle_indices().len() > 0 {
            self.underlying_mle
                .mle_indices()
                .into_iter()
                .map(|index| index.val().unwrap())
                .collect_vec()
        } else {
            vec![]
        };
        let blinding_factor = &C::Scalar::random(blinding_rng);
        let claim_commit = scalar_committer
            .committed_scalar(&self.underlying_mle.bookkeeping_table()[0], blinding_factor);
        HyraxClaim {
            point: claim_chal,
            mle_enum: Some(self.underlying_mle.clone()),
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
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
        blinding_rng: &mut impl Rng,
        scalar_committer: &PedersenCommitter<C>,
    ) -> (Self, HyraxClaim<C::Scalar, CommittedScalar<C>>) {
        // Fix variable on the output layer in order to generate the claim on the previous layer
        output_layer.fix_variable_on_challenge(transcript);
        let committed_claim = output_layer.get_claim(blinding_rng, scalar_committer);
        let commitment = committed_claim.to_claim_commitment().evaluation;
        // Add the commitment to the transcript
        append_x_y_to_transcript_single(
            &commitment,
            transcript,
            "output layer commit x",
            "output layer commit y",
        );

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
        layer_desc: &OutputLayerDescription,
        transcript: &mut impl Transcript<C::Scalar, C::Base>,
    ) -> HyraxClaim<C::Scalar, C> {
        // Get the first set of challenges needed for the output layer.
        let bindings = (0..layer_desc.num_vars)
            .map(|_| {
                let challenge = transcript
                    .get_scalar_field_challenge("output claim point")
                    .unwrap();
                challenge
            })
            .collect_vec();

        // Add the commitment to the transcript
        append_x_y_to_transcript_single(
            &proof.claim_commitment,
            transcript,
            "output layer commit x",
            "output layer commit y",
        );
        HyraxClaim {
            point: bindings,
            mle_enum: None,
            to_layer_id: layer_desc.layer_id,
            evaluation: proof.claim_commitment,
        }
    }
}

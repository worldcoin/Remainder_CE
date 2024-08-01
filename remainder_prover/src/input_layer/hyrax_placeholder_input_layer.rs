//! An input layer to serve as a placeholder for a Hyrax input layer, whose claims are committed to with Pedersen commitments.

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_enum::MleEnum},
};

use super::{InputLayer, InputLayerError, MleInputLayer, VerifierInputLayer};

/// An Input Layer in which the data is sent to the verifier as a placeholder
/// representing a Hyrax input layer that does not have a precommitment.
#[derive(Debug)]
pub struct HyraxPlaceholderInputLayer<F: FieldExt> {
    /// The MLE that the input layer is committing to.
    pub mle: MultilinearExtension<F>,
    pub(crate) layer_id: LayerId,
}

/// The verifier version of the [HyraxPlaceholderInputLayer], which is
/// also a placeholder.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierHyraxPlaceholderInputLayer<F: FieldExt> {
    layer_id: LayerId,
    num_bits: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> VerifierHyraxPlaceholderInputLayer<F> {
    /// To be used only for internal testing!
    /// Generates a new [VerifierPublicInputLayer] from given raw data.
    /// Normally, such a layer would be produced through the
    /// `PublicInputLayer::into_verifier_public_input_layer()` method.
    pub(crate) fn new_raw(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> InputLayer<F> for HyraxPlaceholderInputLayer<F> {
    type Commitment = Vec<F>;

    type VerifierInputLayer = VerifierHyraxPlaceholderInputLayer<F>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.get_evals_vector().clone())
    }

    fn append_commitment_to_transcript(
        _commitment: &Self::Commitment,
        _transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        unimplemented!()
    }

    fn open(
        &self,
        _: &mut impl ProverTranscript<F>,
        _: crate::claims::Claim<F>,
    ) -> Result<(), super::InputLayerError> {
        unimplemented!()
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }

    fn into_verifier_input_layer(&self) -> Self::VerifierInputLayer {
        let num_bits = self.mle.num_vars();

        Self::VerifierInputLayer::new_raw(self.layer_id, num_bits)
    }
}

impl<F: FieldExt> MleInputLayer<F> for HyraxPlaceholderInputLayer<F> {
    fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }
}

impl<F: FieldExt> VerifierInputLayer<F> for VerifierHyraxPlaceholderInputLayer<F> {
    type Commitment = Vec<F>;

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_commitment_from_transcript(
        &self,
        _transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError> {
        unimplemented!()
    }

    fn verify(
        &self,
        _commitment: &Self::Commitment,
        _claim: Claim<F>,
        _: &mut impl VerifierTranscript<F>,
    ) -> Result<(), super::InputLayerError> {
        unimplemented!()
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for HyraxPlaceholderInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        _claim_vecs: &[Vec<F>],
        _claimed_vals: &[F],
        _claimed_mles: Vec<MleEnum<F>>,
        _num_claims: usize,
        _num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        unimplemented!()
    }
}

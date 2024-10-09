//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::ProverTranscript,
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::YieldWLXEvals},
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use super::{
    get_wlx_evaluations_helper, InputLayerTrait,
};

/// An Input Layer in which the data is sent to the verifier
/// "in the clear" (i.e. without a commitment).
#[derive(Debug, Clone)]
pub struct PublicInputLayer<F: Field> {
    mle: MultilinearExtension<F>,
    pub(crate) layer_id: LayerId,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(bound = "F: Field")]
/// The circuit description of a [PublicInputLayer] which stores
/// the shape information of this input layer.
pub struct PublicInputLayerDescription<F: Field> {
    layer_id: LayerId,
    num_bits: usize,
    _marker: PhantomData<F>,
}

impl<F: Field> PublicInputLayerDescription<F> {
    /// Constructor for the [PublicInputLayerDescription] using the layer_id
    /// and the number of variables in the MLE we are storing in the
    /// input layer.
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> PublicInputLayer<F> {
    /// FIXME(Ben) document or remove - could make self.mle public instead?
    pub fn get_evaluations_as_vec(&self) -> &Vec<F> {
        self.mle.get_evals_vector()
    }
}

impl<F: Field> InputLayerTrait<F> for PublicInputLayer<F> {
    type ProverCommitment = Vec<F>;
    type VerifierCommitment = Vec<F>;

    // FIXME(Ben) this function will be redundant
    fn commit(&mut self) -> Result<Self::VerifierCommitment, super::InputLayerError> {
        // Because this is a public input layer, we do not need to commit to the
        // MLE and the "commitment" is just the MLE evaluations themselves.
        Ok(self.mle.get_evals_vector().clone())
    }

    /// Append the commitment to the Fiat-Shamir transcript.
    fn append_commitment_to_transcript(
        commitment: &Self::VerifierCommitment,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        transcript_writer.append_elements("Public Input Commitment", commitment);
    }

    /// We do not have an opening proof because we did not commit to anything.
    /// The MLE exists in the clear.
    fn open(
        &self,
        _: &mut impl ProverTranscript<F>,
        _: crate::claims::Claim<F>,
    ) -> Result<(), super::InputLayerError> {
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }
}

impl<F: Field> PublicInputLayer<F> {
    /// Constructor for the [PublicInputLayer] using the MLE in the input
    /// and the layer_id.
    pub fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }
}

impl<F: Field> YieldWLXEvals<F> for PublicInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<DenseMle<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        get_wlx_evaluations_helper(
            self.mle.clone(),
            claim_vecs,
            claimed_vals,
            claimed_mles,
            num_claims,
            num_idx,
        )
    }
}
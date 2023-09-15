//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};

use crate::{
    layer::{claims::Claim, LayerId},
    mle::{dense::DenseMle, MleRef},
};

use super::{enum_input_layer::InputLayerEnum, InputLayer, InputLayerError};

pub struct RandomInputLayer<F: FieldExt, Tr> {
    mle: Vec<F>,
    pub(crate) layer_id: LayerId,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for RandomInputLayer<F, Tr> {
    type Transcript = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.clone())
    }

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut Self::Transcript,
    ) -> Result<(), TranscriptError> {
        for challenge in commitment {
            let real_chal = transcript.get_challenge("Getting RandomInput")?;
            if *challenge != real_chal {
                return Err(TranscriptError::TranscriptMatchError);
            }
        }
        Ok(())
    }

    fn open(
        &self,
        transcript: &mut Self::Transcript,
        claim: Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), super::InputLayerError> {
        let mut mle_ref =
            DenseMle::<F, F>::new_from_raw(commitment.to_vec(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }

            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            Claim::new_raw(vec![], mle_ref.bookkeeping_table[0])
        };

        if eval == claim {
            Ok(())
        } else {
            Err(InputLayerError::PublicInputVerificationFailed)
        }
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id.clone(), None)
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript> {
        InputLayerEnum::RandomInputLayer(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> RandomInputLayer<F, Tr> {
    ///Generates a random MLE of size `size` that is generated from the FS Transcript
    pub fn new(transcript: &mut Tr, size: usize, layer_id: LayerId) -> Self {
        let mle = transcript
            .get_challenges("Getting Random Challenges", size)
            .unwrap();
        Self {
            mle,
            layer_id,
            _marker: PhantomData,
        }
    }

    pub fn get_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id.clone(), None)
    }
}

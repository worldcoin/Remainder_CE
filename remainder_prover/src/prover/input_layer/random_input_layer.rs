//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{FieldExt, transcript::{Transcript, TranscriptError}};

use crate::{mle::{dense::DenseMle, MleRef}, layer::{Claim, LayerId}};

use super::{InputLayer, InputLayerError, enum_input_layer::InputLayerEnum};

pub struct RandomInputLayer<F: FieldExt, Tr> {
    mle: Vec<F>,
    layer_id: LayerId,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for RandomInputLayer<F, Tr> {
    type Transcript = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.clone())
    }

    fn append_commitment_to_transcript(commitment: &Self::Commitment, transcript: &mut Self::Transcript) -> Result<(), TranscriptError> {
        for challenge in commitment {
            let real_chal = transcript.get_challenge("Getting RandomInput")?;
            if *challenge != real_chal {
                return Err(TranscriptError::TranscriptMatchError)
            }
        }
        Ok(())
    }

    fn open(&self, transcript: &mut Self::Transcript, claim: Claim<F>) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F, F>::new_from_raw(commitment.to_vec(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let mut curr_bit = 0;
        let mut eval = None;
        for &chal in claim.0.iter() {
            if chal != F::one() && chal != F::zero() {
                eval = mle_ref.fix_variable(curr_bit, chal);
                curr_bit += 1;
            }
        }

        let eval = eval.ok_or(InputLayerError::PublicInputVerificationFailed)?;

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
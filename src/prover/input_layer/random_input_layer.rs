//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use lcpc_2d::{FieldExt, fs_transcript::halo2_remainder_transcript::{Transcript, TranscriptError}};

use crate::{mle::{dense::DenseMle, MleRef}, layer::{Claim, LayerId}};

use super::{InputLayer, InputLayerError};

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
        Ok(())
    }

    fn open(&self, transcript: &mut Self::Transcript, claim: Claim<F>) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F, F>::new_from_raw(commitment.to_vec(), LayerId::Input(0), None).mle_ref();

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
}
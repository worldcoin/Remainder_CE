//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
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

impl<F: FieldExt, Tr: TranscriptSponge<F>> InputLayer<F> for RandomInputLayer<F, Tr> {
    type Sponge = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.clone())
    }

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, Self::Sponge>,
    ) -> Result<(), InputLayerError> {
        for challenge in commitment {
            let real_chal = transcript_reader
                .get_challenge("Getting RandomInput")
                .map_err(|e| InputLayerError::TranscriptError(e))?;
            if *challenge != real_chal {
                return Err(InputLayerError::TranscriptMatchError);
            }
        }
        Ok(())
    }

    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    ) {
        unimplemented!()
    }

    fn open(
        &self,
        _transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
        _claim: Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(
        commitment: &Self::Commitment,
        _opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        _transcript_reader: &mut TranscriptReader<F, Self::Sponge>,
    ) -> Result<(), super::InputLayerError> {
        // println!("3, calling verify");
        let mut mle_ref =
            DenseMle::<F, F>::new_from_raw(commitment.to_vec(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            // println!("1, eval = {:#?}, claim = {:#?}", eval, claim);

            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            Claim::new_raw(vec![], mle_ref.current_mle[0])
        };

        if eval.get_point() == claim.get_point() && eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            // println!("2, eval = {:#?}, claim = {:#?}", eval, claim);
            Err(InputLayerError::PublicInputVerificationFailed)
        }
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id, None)
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Sponge> {
        InputLayerEnum::RandomInputLayer(self)
    }
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> RandomInputLayer<F, Tr> {
    ///Generates a random MLE of size `size` that is generated from the FS Transcript
    pub fn new(
        transcript_writer: &mut TranscriptWriter<F, Tr>,
        size: usize,
        layer_id: LayerId,
    ) -> Self {
        let mle = transcript_writer.get_challenges("Getting Random Challenges", size);
        Self {
            mle,
            layer_id,
            _marker: PhantomData,
        }
    }

    pub fn get_mle(&self) -> DenseMle<F, F> {
        DenseMle::new_from_raw(self.mle.clone(), self.layer_id, None)
    }
}

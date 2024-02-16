//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

use crate::{
    layer::{claims::Claim, LayerId},
    mle::{dense::DenseMle, MleRef},
};

use super::{enum_input_layer::InputLayerEnum, InputLayer, InputLayerError, MleInputLayer};

///An Input Layer that is send to the verifier in the clear
pub struct PublicInputLayer<F: FieldExt, Tr> {
    mle: DenseMle<F, F>,
    pub(crate) layer_id: LayerId,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> InputLayer<F> for PublicInputLayer<F, Tr> {
    type Sponge = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.mle.clone())
    }

    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, Self::Sponge>,
    ) {
        transcript_writer.append_elements("Public Input Commitment", commitment);
    }

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, Self::Sponge>,
    ) -> Result<(), InputLayerError> {
        let num_elements = commitment.len();
        let transcript_commitment = transcript_reader
            .consume_elements("Public Input Commitment", num_elements)
            .map_err(|e| InputLayerError::TranscriptError(e))?;
        debug_assert_eq!(transcript_commitment, commitment);
        Ok(())
    }

    fn open(
        &self,
        _: &mut TranscriptWriter<F, Self::Sponge>,
        _: crate::layer::claims::Claim<F>,
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
            DenseMle::<F, F>::new_from_raw(commitment.clone(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            debug_assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            // println!("1, eval = {:#?}, claim = {:#?}", eval, claim);
            // dbg!(&eval);
            // dbg!(&claim);
            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            Claim::new_raw(vec![], mle_ref.bookkeeping_table[0])
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
        self.mle.clone()
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Sponge> {
        InputLayerEnum::PublicInputLayer(self)
    }
}

impl<F: FieldExt, Tr: TranscriptSponge<F>> MleInputLayer<F> for PublicInputLayer<F, Tr> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            _marker: PhantomData,
        }
    }
}

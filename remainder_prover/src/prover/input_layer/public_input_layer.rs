//! An input layer that is sent to the verifier in the clear

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};

use crate::{
    layer::{Claim, LayerId},
    mle::{dense::DenseMle, MleRef},
};

use super::{enum_input_layer::InputLayerEnum, InputLayer, InputLayerError, MleInputLayer};

///An Input Layer that is send to the verifier in the clear
pub struct PublicInputLayer<F: FieldExt, Tr> {
    mle: DenseMle<F, F>,
    pub(crate) layer_id: LayerId,
    _marker: PhantomData<Tr>,
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for PublicInputLayer<F, Tr> {
    type Transcript = Tr;

    type Commitment = Vec<F>;

    type OpeningProof = ();

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        Ok(self.mle.mle.clone())
    }

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut Self::Transcript,
    ) -> Result<(), TranscriptError> {
        transcript.append_field_elements("Public Input Commitment", &commitment)
    }

    fn open(
        &self,
        _: &mut Self::Transcript,
        _: crate::layer::Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        Ok(())
    }

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: Claim<F>, transcript: &mut Self::Transcript) -> Result<(), super::InputLayerError> {
        let mut mle_ref = DenseMle::<F, F>::new_from_raw(commitment.clone(), LayerId::Input(0), None).mle_ref();
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_vars != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.0.iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            debug_assert_eq!(mle_ref.bookkeeping_table().len(), 1);
            dbg!(&eval);
            dbg!(&claim);
            eval.ok_or(InputLayerError::PublicInputVerificationFailed)?
        } else {
            (vec![], mle_ref.bookkeeping_table[0])
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
        self.mle.clone()
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript> {
        InputLayerEnum::PublicInputLayer(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> MleInputLayer<F> for PublicInputLayer<F, Tr> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            _marker: PhantomData,
        }
    }
}

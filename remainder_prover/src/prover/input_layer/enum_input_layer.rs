//! A wrapper type that makes working with variants of InputLayer easier

use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    layer::{Claim, LayerId},
    mle::dense::DenseMle,
};

use super::{
    ligero_input_layer::{LigeroCommitment, LigeroInputLayer, LigeroInputProof},
    public_input_layer::PublicInputLayer,
    random_input_layer::RandomInputLayer,
    InputLayer,
};

///A wrapper type that makes working with variants of InputLayer easier
pub enum InputLayerEnum<F: FieldExt, Tr> {
    LigeroInputLayer(LigeroInputLayer<F, Tr>),
    PublicInputLayer(PublicInputLayer<F, Tr>),
    RandomInputLayer(RandomInputLayer<F, Tr>),
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub enum CommitmentEnum<F: FieldExt> {
    LigeroCommitment(LigeroCommitment<F>),
    PublicCommitment(Vec<F>),
    RandomCommitment(Vec<F>),
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub enum OpeningEnum<F: FieldExt> {
    LigeroProof(LigeroInputProof<F>),
    PublicProof(()),
    RandomProof(()),
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for InputLayerEnum<F, Tr> {
    type Transcript = Tr;

    type Commitment = CommitmentEnum<F>;

    type OpeningProof = OpeningEnum<F>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => {
                Ok(CommitmentEnum::LigeroCommitment(layer.commit()?))
            }
            InputLayerEnum::PublicInputLayer(layer) => {
                Ok(CommitmentEnum::PublicCommitment(layer.commit()?))
            }
            InputLayerEnum::RandomInputLayer(layer) => {
                Ok(CommitmentEnum::RandomCommitment(layer.commit()?))
            }
        }
    }

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut Self::Transcript,
    ) -> Result<(), TranscriptError> {
        match commitment {
            CommitmentEnum::LigeroCommitment(commit) => {
                LigeroInputLayer::<F, Tr>::append_commitment_to_transcript(commit, transcript)
            }
            CommitmentEnum::PublicCommitment(commit) => {
                PublicInputLayer::append_commitment_to_transcript(commit, transcript)
            }
            CommitmentEnum::RandomCommitment(commit) => {
                RandomInputLayer::append_commitment_to_transcript(commit, transcript)
            }
        }
    }

    fn open(
        &self,
        transcript: &mut Self::Transcript,
        claim: crate::layer::Claim<F>,
    ) -> Result<Self::OpeningProof, super::InputLayerError> {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => {
                Ok(OpeningEnum::LigeroProof(layer.open(transcript, claim)?))
            }
            InputLayerEnum::PublicInputLayer(layer) => {
                Ok(OpeningEnum::PublicProof(layer.open(transcript, claim)?))
            }
            InputLayerEnum::RandomInputLayer(layer) => {
                Ok(OpeningEnum::RandomProof(layer.open(transcript, claim)?))
            }
        }
    }

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), super::InputLayerError> {
        match commitment {
            CommitmentEnum::LigeroCommitment(commit) => {
                if let OpeningEnum::LigeroProof(opening_proof) = opening_proof {
                    LigeroInputLayer::<F, Tr>::verify(commit, opening_proof, claim, transcript)
                } else {
                    panic!()
                }
            }
            CommitmentEnum::PublicCommitment(commit) => {
                if let OpeningEnum::PublicProof(opening_proof) = opening_proof {
                    PublicInputLayer::verify(commit, opening_proof, claim, transcript)
                } else {
                    panic!()
                }
            }
            CommitmentEnum::RandomCommitment(commit) => {
                if let OpeningEnum::RandomProof(opening_proof) = opening_proof {
                    RandomInputLayer::verify(commit, opening_proof, claim, transcript)
                } else {
                    panic!()
                }
            }
        }
    }

    fn layer_id(&self) -> &LayerId {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.layer_id(),
            InputLayerEnum::PublicInputLayer(layer) => layer.layer_id(),
            InputLayerEnum::RandomInputLayer(layer) => layer.layer_id(),
        }
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.get_padded_mle(),
            InputLayerEnum::PublicInputLayer(layer) => layer.get_padded_mle(),
            InputLayerEnum::RandomInputLayer(layer) => layer.get_padded_mle(),
        }
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript> {
        self
    }
}

impl<F: FieldExt, Tr: Transcript<F>> InputLayerEnum<F, Tr> {
    pub fn set_layer_id(&mut self, layer_id: LayerId) {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::PublicInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::RandomInputLayer(layer) => layer.layer_id = layer_id,
        }
    }
}

//! A wrapper type that makes working with variants of InputLayer easier.

use remainder_shared_types::{transcript::VerifierTranscript, Field};
use serde::{Deserialize, Serialize};

use crate::{input_layer_enum, layer::LayerId, mle::evals::MultilinearExtension};

use super::{
    hyrax_input_layer::HyraxInputLayerDescription,
    ligero_input_layer::{LigeroInputLayer, LigeroInputLayerDescription},
    public_input_layer::{PublicInputLayer, PublicInputLayerDescription},
    CommitmentEnum, InputLayer, InputLayerDescription, InputLayerError,
};

#[derive(Serialize, Deserialize, Debug, Hash)]
#[serde(bound = "F: Field")]
/// An enum representing the different types of descriptions for each layer,
/// each description containing the shape information of the corresponding layer.
pub enum InputLayerDescriptionEnum<F: Field> {
    /// The circuit description for a public input layer.
    PublicInputLayer(PublicInputLayerDescription<F>),
    /// The circuit description for a ligero input layer.
    LigeroInputLayer(LigeroInputLayerDescription<F>),
    /// The circuit description for a hyrax input layer.
    HyraxInputLayer(HyraxInputLayerDescription<F>),
}

impl<F: Field> InputLayerDescription<F> for InputLayerDescriptionEnum<F> {
    type Commitment = InputLayerEnumVerifierCommitment<F>;

    fn layer_id(&self) -> LayerId {
        match self {
            InputLayerDescriptionEnum::PublicInputLayer(circuit_public_input_layer) => {
                circuit_public_input_layer.layer_id()
            }
            InputLayerDescriptionEnum::LigeroInputLayer(circuit_ligero_input_layer) => {
                circuit_ligero_input_layer.layer_id()
            }
            InputLayerDescriptionEnum::HyraxInputLayer(circuit_hyrax_input_layer) => {
                circuit_hyrax_input_layer.layer_id
            }
        }
    }

    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError> {
        match self {
            InputLayerDescriptionEnum::PublicInputLayer(circuit_public_input_layer) => {
                Ok(InputLayerEnumVerifierCommitment::PublicInputLayer(
                    circuit_public_input_layer
                        .get_commitment_from_transcript(transcript_reader)
                        .unwrap(),
                ))
            }
            InputLayerDescriptionEnum::LigeroInputLayer(circuit_ligero_input_layer) => {
                Ok(InputLayerEnumVerifierCommitment::LigeroInputLayer(
                    circuit_ligero_input_layer
                        .get_commitment_from_transcript(transcript_reader)
                        .unwrap(),
                ))
            }
            InputLayerDescriptionEnum::HyraxInputLayer(_circuit_hyrax_input_layer) => {
                panic!("The circuit input layer trait is not implemented for hyrax input layers!")
            }
        }
    }

    fn convert_into_prover_input_layer(
        &self,
        mle: MultilinearExtension<F>,
        precommit: &Option<CommitmentEnum<F>>,
    ) -> InputLayerEnum<F> {
        match self {
            InputLayerDescriptionEnum::PublicInputLayer(circuit_public_input_layer) => {
                circuit_public_input_layer.convert_into_prover_input_layer(mle, precommit)
            }
            InputLayerDescriptionEnum::LigeroInputLayer(circuit_ligero_input_layer) => {
                circuit_ligero_input_layer.convert_into_prover_input_layer(mle, precommit)
            }
            InputLayerDescriptionEnum::HyraxInputLayer(_circuit_hyrax_input_layer) => {
                panic!("The circuit input layer trait is not implemented for hyrax input layers!")
            }
        }
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: crate::claims::RawClaim<F>,
        transcript_reader: &mut impl remainder_shared_types::transcript::VerifierTranscript<F>,
    ) -> Result<(), super::InputLayerError> {
        match self {
            InputLayerDescriptionEnum::PublicInputLayer(circuit_public_input_layer) => {
                match commitment {
                    InputLayerEnumVerifierCommitment::PublicInputLayer(public_commitment) => {
                        circuit_public_input_layer.verify(
                            public_commitment,
                            claim,
                            transcript_reader,
                        )
                    }
                    _ => panic!("wrong commitment type for input layer description!"),
                }
            }
            InputLayerDescriptionEnum::LigeroInputLayer(circuit_ligero_input_layer) => {
                match commitment {
                    InputLayerEnumVerifierCommitment::LigeroInputLayer(ligero_commitment) => {
                        circuit_ligero_input_layer.verify(
                            ligero_commitment,
                            claim,
                            transcript_reader,
                        )
                    }
                    _ => panic!("wrong commitment type for input layer description!"),
                }
            }
            InputLayerDescriptionEnum::HyraxInputLayer(_circuit_hyrax_input_layer) => {
                panic!("Hyrax input layer is not supported by this trait!")
            }
        }
    }
}

input_layer_enum!(
    InputLayerEnum,
    (LigeroInputLayer: LigeroInputLayer<F>),
    (PublicInputLayer: PublicInputLayer<F>)
);

impl<F: Field> InputLayerEnum<F> {
    /// This function sets the layer ID of the corresponding input layer.
    pub fn set_layer_id(&mut self, layer_id: LayerId) {
        match self {
            InputLayerEnum::LigeroInputLayer(layer) => layer.layer_id = layer_id,
            InputLayerEnum::PublicInputLayer(layer) => layer.layer_id = layer_id,
        }
    }
}

use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt,
};

use std::fmt::Debug;
use serde::{Deserialize, Serialize};

use crate::layer::{GKRLayer, Layer};

use super::input_layer::{InputLayer, ligero_input_layer::LigeroInputLayer, public_input_layer::PublicInputLayer};

#[macro_export]
///This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType))
macro_rules! layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[derive(serde::Serialize, serde::Deserialize, Clone)]
        #[serde(bound = "F: FieldExt")]
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: FieldExt> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name($variant),
            )*
        }

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: FieldExt")]    
            pub enum [<$type_name Proof>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Proof variant"]
                    $var_name(<$variant as Layer<F>>::Proof),
                )*
            }
        }

        impl<F: FieldExt> $crate::layer::Layer<F> for $type_name<F> {
            paste::paste! { type Proof = [<$type_name Proof>]<F>;}
            fn prove_rounds(
                &mut self,
                claim: $crate::layer::claims::Claim<F>,
                transcript: &mut impl remainder_shared_types::transcript::Transcript<F>,
            ) -> Result<Self::Proof, super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::Proof::$var_name(layer.prove_rounds(claim, transcript)?)),
                    )*
                }
            }

            fn verify_rounds(
                &mut self,
                claim: $crate::layer::claims::Claim<F>,
                proof: Self::Proof,
                transcript: &mut impl remainder_shared_types::transcript::Transcript<F>,
            ) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => {
                            let proof = match proof {
                                Self::Proof::$var_name(proof) => proof,
                                _ => unreachable!()
                            };
                            layer.verify_rounds(claim, proof, transcript)
                        },
                    )*
                }
            }

            fn get_claims(&self) -> Result<Vec<$crate::layer::claims::Claim<F>>, super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.get_claims(),
                    )*
                }
            }

            fn id(&self) -> &super::LayerId {
                match self {
                    $(
                        Self::$var_name(layer) => layer.id(),
                    )*
                }
            }

            fn get_wlx_evaluations(
                &self,
                claim_vecs: &Vec<Vec<F>>,
                claimed_vals: &Vec<F>,
                claimed_mles: Vec<$crate::mle::mle_enum::MleEnum<F>>,
                num_claims: usize,
                num_idx: usize,
            ) -> Result<Vec<F>, $crate::layer::claims::ClaimError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.get_wlx_evaluations(claim_vecs, claimed_vals, claimed_mles, num_claims, num_idx),
                    )*
                }
            }

        }

        $(
            impl<F: FieldExt> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(var)
                }
            }
        )*
    }
}

#[macro_export]
macro_rules! input_layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: FieldExt> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name($variant),
            )*
        }

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound = "F: FieldExt")]    
            pub enum [<$type_name Commitment>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::Commitment),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound = "F: FieldExt")]    
            pub enum [<$type_name OpeningProof>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::OpeningProof),
                )*
            }
        }

        impl<F: FieldExt> $crate::prover::InputLayer<F> for $type_name<F> {
            paste::paste! {
                type Commitment = [<$type_name Commitment>]<F>;
                type OpeningProof = [<$type_name OpeningProof>]<F>;
            }

            fn commit(&mut self) -> Result<Self::Commitment, $crate::prover::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::Commitment::$var_name(layer.commit()?)),
                    )*
                }
            }

            fn append_commitment_to_transcript(
                commitment: &Self::Commitment,
                transcript: &mut impl Transcript<F>,
            ) -> Result<(), remainder_shared_types::transcript::TranscriptError> {
                match commitment {
                    $(
                        Self::Commitment::$var_name(commitment) => <$variant as InputLayer<F>>::append_commitment_to_transcript(commitment, transcript),
                    )*
                }
            }

            fn open(
                &self,
                transcript: &mut impl Transcript<F>,
                claim: $crate::prover::Claim<F>,
            ) -> Result<Self::OpeningProof, $crate::prover::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::OpeningProof::$var_name(layer.open(transcript, claim)?)),
                    )*
                }
            }

            fn verify(
                commitment: &Self::Commitment,
                opening_proof: &Self::OpeningProof,
                claim: $crate::prover::Claim<F>,
                transcript: &mut impl Transcript<F>,
            ) -> Result<(), $crate::prover::InputLayerError> {
                match commitment {
                    $(
                        Self::Commitment::$var_name(commitment) => {
                            if let Self::OpeningProof::$var_name(opening_proof) = opening_proof {
                                <$variant as InputLayer<F>>::verify(commitment, opening_proof, claim, transcript)
                            } else {
                                unreachable!()
                            }
                        },
                    )*
                }
            }

            fn layer_id(&self) -> &$crate::prover::LayerId {
                match self {
                    $(
                        Self::$var_name(layer) => layer.layer_id(),
                    )*
                }
            }

            fn get_padded_mle(&self) -> $crate::prover::DenseMle<F, F>{
                match self {
                    $(
                        Self::$var_name(layer) => layer.get_padded_mle(),
                    )*
                }
            }

        }

        $(
            impl<F: FieldExt> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(var)
                }
            }
        )*
    }
}

///A trait for bundling a group of types that define the interfaces that go into a GKR Prover
pub trait ProofSystem<F: FieldExt> {
    ///A trait that defines the allowed Layer for this ProofSystem
    type Layer: Layer<F> + Serialize + for<'a> Deserialize<'a> + Debug;
    
    ///A trait that defines the allowed InputLayer for this ProofSystem
    type InputLayer: InputLayer<F>;

    ///The Transcript this proofsystem uses for F-S
    type Transcript: Transcript<F>;
}

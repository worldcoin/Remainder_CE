use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonSponge, TranscriptSponge},
    FieldExt,
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::{
    claims::{wlx_eval::WLXAggregator, ClaimAggregator, YieldClaim},
    layer::{layer_enum::LayerEnum, Layer},
    mle::{mle_enum::MleEnum, Mle},
};

use crate::input_layer::{enum_input_layer::InputLayerEnum, InputLayer};

#[macro_export]
///This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType), ..)
macro_rules! layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
        #[serde(bound = "F: FieldExt")]
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: FieldExt> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name(Box<$variant>),
            )*
        }

        paste::paste! {
            /*
            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: FieldExt")]
            #[doc = r"Remainder generated Proof enum"]
            pub enum [<$type_name Proof>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Proof variant"]
                    $var_name(<$variant as Layer<F>>::Proof),
                )*
            }
            */

            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: FieldExt")]
            #[doc = r"Verfier layer description enum"]
            pub enum [<Verifier$type_name>]<F: FieldExt> {
                $(
                    #[doc = "Verifier layer description variant"]
                    $var_name(<$variant as Layer<F>>::VerifierLayer),
                )*
            }

            impl<F: FieldExt> $crate::layer::VerifierLayer<F> for [<Verifier$type_name>]<F> {
                fn id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.id(),
                        )*
                    }
                }

                fn verify_rounds(
                    &self,
                    claim: $crate::claims::Claim<F>,
                    transcript: &mut $crate::remainder_shared_types::transcript::TranscriptReader<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>,
                ) -> Result<(), super::VerificationError> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.verify_rounds(claim, transcript),
                        )*
                    }
                }

            }
        }

        impl<F: FieldExt> $crate::layer::Layer<F> for $type_name<F> {
            paste::paste! {
                // type Proof = [<$type_name Proof>]<F>;
                type VerifierLayer = [<Verifier $type_name>]<F>;
            }

            fn id(&self) -> super::LayerId {
                match self {
                    $(
                        Self::$var_name(layer) => layer.id(),
                    )*
                }
            }

            fn prove_rounds(
                &mut self,
                claim: $crate::claims::Claim<F>,
                transcript: &mut $crate::remainder_shared_types::transcript::TranscriptWriter<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>,
            ) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.prove_rounds(claim, transcript),
                    )*
                }
            }
        }

        $(
            impl<F: FieldExt> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*
    }
}

#[macro_export]
///This macro generates an inputlayer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement InputLayer, and the enum will also implement InputLayer and pass methods to it's variants
///
/// Usage:
///
/// input_layer_enum(EnumName, (FirstVariant: InputLayerType), (SecondVariant: SecondInputLayerType), ..)
macro_rules! input_layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: FieldExt> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name(Box<$variant>),
            )*
        }

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound = "F: FieldExt")]
            #[doc = r"Remainder generated commitment enum"]
            pub enum [<$type_name Commitment>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::Commitment),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound = "F: FieldExt")]
            #[doc = r"Remainder generated opening proof enum"]
            pub enum [<$type_name OpeningProof>]<F: FieldExt> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::OpeningProof),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize)]
            #[serde(bound = "F: FieldExt")]
            #[doc = r"Verifier layer description enum"]
            pub enum [<$type_name VerifierInputLayer>]<F: FieldExt> {
                $(
                    #[doc = "Verifier layer description variant"]
                    $var_name(<$variant as InputLayer<F>>::VerifierInputLayer),
                )*
            }
        }

        impl<F: FieldExt> $crate::input_layer::InputLayer<F> for $type_name<F> {
            paste::paste! {
                type Commitment = [<$type_name Commitment>]<F>;
                type OpeningProof = [<$type_name OpeningProof>]<F>;
                type VerifierInputLayer = [<$type_name VerifierInputLayer>]<F>;
            }

            fn commit(&mut self) -> Result<Self::Commitment, $crate::input_layer::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::Commitment::$var_name(layer.commit()?)),
                    )*
                }
            }

            fn prover_append_commitment_to_transcript(
                commitment: &Self::Commitment,
                transcript: &mut $crate::remainder_shared_types::transcript::TranscriptWriter<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>,
            ) {
                match commitment {
                    $(
                        Self::Commitment::$var_name(commitment) => <$variant as InputLayer<F>>::prover_append_commitment_to_transcript(commitment, transcript),
                    )*
                }
            }

            fn verifier_get_commitment_from_transcript(
                transcript: &mut $crate::remainder_shared_types::transcript::TranscriptReader<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>
            ) -> Result<Self::Commitment, $crate::input_layer::InputLayerError> {
                todo!()
                /*
                match commitment {
                    $(
                        Self::Commitment::$var_name(commitment) => <$variant as InputLayer<F>>::verifier_get_commitment_from_transcript(transcript),
                    )*
                }
                */
            }

            fn open(
                &self,
                transcript: &mut $crate::remainder_shared_types::transcript::TranscriptWriter<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>,
                claim: $crate::claims::Claim<F>,
            ) -> Result<Self::OpeningProof, $crate::input_layer::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::OpeningProof::$var_name(layer.open(transcript, claim)?)),
                    )*
                }
            }

            fn verify(
                commitment: &Self::Commitment,
                opening_proof: &Self::OpeningProof,
                claim: $crate::claims::Claim<F>,
                transcript: &mut $crate::remainder_shared_types::transcript::TranscriptReader<F, impl $crate::remainder_shared_types::transcript::TranscriptSponge<F>>,
            ) -> Result<(), $crate::input_layer::InputLayerError> {
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

            fn layer_id(&self) -> &$crate::layer::LayerId {
                match self {
                    $(
                        Self::$var_name(layer) => layer.layer_id(),
                    )*
                }
            }

            fn get_padded_mle(&self) -> $crate::mle::dense::DenseMle<F,>{
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
                    Self::$var_name(Box::new(var))
                }
            }
        )*
    }
}

/// A trait for bundling a group of types that define the interfaces that go
/// into a GKR Prover.
pub trait ProofSystem<F: FieldExt> {
    /// A trait that defines the allowed Layer for this ProofSystem.
    type Layer: Layer<F>
        + Serialize
        + for<'a> Deserialize<'a>
        + Debug
        + YieldClaim<F, <Self::ClaimAggregator as ClaimAggregator<F>>::Claim>;

    /// A trait that defines the allowed InputLayer for this ProofSystem.
    type InputLayer: InputLayer<F>;

    /// The Transcript this proofsystem uses for Fiat-Shamir.
    type Transcript: TranscriptSponge<F>;

    /// The MleRef type that serves as the output layer representation
    type OutputLayer: Mle<F>
        + YieldClaim<F, <Self::ClaimAggregator as ClaimAggregator<F>>::Claim>
        + Serialize
        + for<'de> Deserialize<'de>;

    ///The logic that handles how to aggregate claims
    /// As this trait defines the 'bridge' between layers, some helper traits may be neccessary to implement
    /// on the other layer types
    type ClaimAggregator: ClaimAggregator<F, Layer = Self::Layer, InputLayer = Self::InputLayer>;
}

/// The default proof system for the remainder prover
pub struct DefaultProofSystem;

impl<F: FieldExt> ProofSystem<F> for DefaultProofSystem {
    type Layer = LayerEnum<F>;

    type InputLayer = InputLayerEnum<F>;

    type Transcript = PoseidonSponge<F>;

    type OutputLayer = MleEnum<F>;

    type ClaimAggregator = WLXAggregator<F, Self::Layer, Self::InputLayer>;
}

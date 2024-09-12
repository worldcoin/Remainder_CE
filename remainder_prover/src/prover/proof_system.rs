use remainder_shared_types::{
    transcript::{
        poseidon_transcript::PoseidonSponge, test_transcript::TestSponge, TranscriptSponge,
    },
    Field,
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::layer::Layer;
use crate::{
    claims::{wlx_eval::WLXAggregator, ClaimAggregator, YieldClaim},
    input_layer::VerifierInputLayer,
    layer::{layer_enum::LayerEnum, CircuitLayer},
    output_layer::{mle_output_layer::MleOutputLayer, CircuitOutputLayer, OutputLayer},
};

use crate::input_layer::{enum_input_layer::InputLayerEnum, InputLayer};

///This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType), ..)
#[macro_export]
macro_rules! layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
        #[serde(bound = "F: Field")]
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: Field> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name(Box<$variant>),
            )*
        }

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Circuit layer description enum"]
            pub enum [<Circuit $type_name>]<F: Field> {
                $(
                    #[doc = "Circuit layer description variant"]
                    $var_name(<$variant as crate::layer::Layer<F>>::CircuitLayer),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Verfier layer description enum"]
            pub enum [<Verifier $type_name>]<F: Field> {
                $(
                    #[doc = "Verifier layer description variant"]
                    $var_name(<<$variant as crate::layer::Layer<F>>::CircuitLayer as crate::layer::CircuitLayer<F>>::VerifierLayer),
                )*
            }


            impl<F: Field> $crate::layer::CircuitLayer<F> for [<Circuit$type_name>]<F> {
                type VerifierLayer = [<Verifier $type_name>]<F>;

                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn verify_rounds(
                    &self,
                    claim: $crate::claims::Claim<F>,
                    transcript: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> Result<Self::VerifierLayer, super::VerificationError> {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(Self::VerifierLayer::$var_name(layer.verify_rounds(claim, transcript)?)),
                        )*
                    }
                }

                fn sumcheck_round_indices(
                    &self
                ) -> Vec<usize> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.sumcheck_round_indices(),
                        )*
                    }
                }

                fn into_verifier_layer(
                    &self,
                    sumcheck_bindings: &[F],
                    claim_point: &[F],
                    transcript_reader: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> Result<Self::VerifierLayer, super::VerificationError> {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(Self::VerifierLayer::$var_name(layer.into_verifier_layer(sumcheck_bindings, claim_point, transcript_reader)?)),
                        )*
                    }
                }

                fn get_post_sumcheck_layer(
                    &self,
                    round_challenges: &[F],
                    claim_challenges: &[F],
                ) -> $crate::layer::PostSumcheckLayer<F, Option<F>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_post_sumcheck_layer(round_challenges, claim_challenges),
                        )*
                    }
                }

                fn max_degree(&self) -> usize {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.max_degree(),
                        )*
                    }
                }
            }

            impl<F: Field> $crate::layer::VerifierLayer<F> for [<Verifier$type_name>]<F> {
                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }
            }
        }

        impl<F: Field> $crate::layer::Layer<F> for $type_name<F> {
            paste::paste! {
                type CircuitLayer = [<Circuit $type_name>]<F>;
            }

            fn into_circuit_layer(&self) -> Result<CircuitLayerEnum<F>, LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => Ok(Self::CircuitLayer::$var_name(layer.into_circuit_layer()?)),
                    )*
                }
            }

            fn layer_id(&self) -> super::LayerId {
                match self {
                    $(
                        Self::$var_name(layer) => layer.layer_id(),
                    )*
                }
            }

            fn prove_rounds(
                &mut self,
                claim: $crate::claims::Claim<F>,
                transcript: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<F>,
            ) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.prove_rounds(claim, transcript),
                    )*
                }
            }

            fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.initialize_sumcheck(claim_point),
                    )*
                }
            }

            fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.compute_round_sumcheck_message(round_index),
                    )*
                }
            }

            fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.bind_round_variable(round_index, challenge),
                    )*
                }
            }

            fn sumcheck_round_indices(&self) -> Vec<usize> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.sumcheck_round_indices(),
                    )*
                }
            }

            fn max_degree(&self) -> usize {
                match self {
                    $(
                        Self::$var_name(layer) => layer.max_degree(),
                    )*
                }
            }

            fn get_post_sumcheck_layer(
                &self,
                round_challenges: &[F],
                claim_challenges: &[F],
            ) -> crate::layer::PostSumcheckLayer<F, F> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.get_post_sumcheck_layer(round_challenges, claim_challenges),
                    )*
                }
            }
        }

        $(
            impl<F: Field> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*

        paste::paste! {
            impl<F: Field> YieldClaim<ClaimMle<F>> for [<Verifier $type_name>]<F> {
                fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_claims(),
                        )*
                    }
                }
            }
        }
    }
}

///This macro generates an inputlayer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement InputLayer, and the enum will also implement InputLayer and pass methods to it's variants
///
/// Usage:
///
/// input_layer_enum(EnumName, (FirstVariant: InputLayerType), (SecondVariant: SecondInputLayerType), ..)
#[macro_export]
macro_rules! input_layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {
        #[derive(Debug)]
        #[doc = r"Remainder generated trait enum"]
        pub enum $type_name<F: Field> {
            $(
                #[doc = "Remainder generated layer variant"]
                $var_name(Box<$variant>),
            )*
        }

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Remainder generated commitment enum"]
            pub enum [<$type_name Commitment>]<F: Field> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::Commitment),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Verifier layer description enum"]
            pub enum [<Verifier $type_name>]<F: Field> {
                $(
                    #[doc = "Verifier layer description variant"]
                    $var_name(<$variant as InputLayer<F>>::VerifierInputLayer),
                )*
            }
        }

        impl<F: Field> $crate::input_layer::InputLayer<F> for $type_name<F> {
            paste::paste! {
                type Commitment = [<$type_name Commitment>]<F>;
                type VerifierInputLayer = [<Verifier $type_name>]<F>;
            }

            fn into_verifier_input_layer(&self) -> Self::VerifierInputLayer {
                match self {
                    $(
                        Self::$var_name(layer) => Self::VerifierInputLayer::$var_name(layer.into_verifier_input_layer()),
                    )*
                }
            }

            fn commit(&mut self) -> Result<Self::Commitment, $crate::input_layer::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => {
                            Ok(Self::Commitment::$var_name(layer.commit()?))
                        }
                    )*
                }
            }

            fn append_commitment_to_transcript(
                commitment: &Self::Commitment,
                transcript_writer: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<F>,
            ) {
                match commitment {
                    $(
                        Self::Commitment::$var_name(commitment) => <$variant as InputLayer<F>>::append_commitment_to_transcript(commitment, transcript_writer),
                    )*
                }
            }

            fn open(
                &self,
                transcript_writer: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<F>,
                claim: $crate::claims::Claim<F>,
            ) -> Result<(), $crate::input_layer::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.open(transcript_writer, claim),
                    )*
                }
            }

            fn layer_id(&self) -> $crate::layer::LayerId {
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

        paste::paste! {
            impl<F: Field> $crate::input_layer::VerifierInputLayer<F> for [<Verifier $type_name>]<F> {
                type Commitment = [<$type_name Commitment>]<F>;

                fn layer_id(&self) -> $crate::layer::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn get_commitment_from_transcript(
                    &self,
                    transcript_reader: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                )  -> Result<Self::Commitment, $crate::input_layer::InputLayerError> {
                    match self {
                        $(
                            Self::$var_name(layer) => {
                                let commitment = layer.get_commitment_from_transcript(transcript_reader)?;
                                Ok(Self::Commitment::$var_name(commitment))
                            }
                        )*
                    }
                }

                fn verify(
                    &self,
                    commitment: &Self::Commitment,
                    claim: $crate::claims::Claim<F>,
                    transcript_reader: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> Result<(), $crate::input_layer::InputLayerError> {
                    match self {
                        $(
                            Self::$var_name(layer) => {
                                if let Self::Commitment::$var_name(commitment) = commitment {
                                    layer.verify(commitment, claim, transcript_reader)
                                } else {
                                    unreachable!()
                                }
                            }
                        )*
                    }
                }
            }
        }

        $(
            impl<F: Field> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*
    }
}

/// A trait for bundling a group of types that define the interfaces that go
/// into a GKR Prover.
pub trait ProofSystem<F: Field> {
    /// A trait that defines the allowed Layer for this ProofSystem.
    type Layer: Layer<
            F,
            CircuitLayer: CircuitLayer<
                F,
                VerifierLayer: YieldClaim<<Self::ClaimAggregator as ClaimAggregator<F>>::Claim>,
            >,
        > + Serialize
        + for<'a> Deserialize<'a>
        + Debug
        + YieldClaim<<Self::ClaimAggregator as ClaimAggregator<F>>::Claim>;

    /// A trait that defines the allowed InputLayer for this ProofSystem.
    type InputLayer: InputLayer<F, VerifierInputLayer: VerifierInputLayer<F>> + Debug;

    /// The Transcript this proofsystem uses for Fiat-Shamir.
    type Transcript: TranscriptSponge<F>;

    /// The MleRef type that serves as the output layer representation
    type OutputLayer: OutputLayer<
            F,
            CircuitOutputLayer: CircuitOutputLayer<
                F,
                VerifierOutputLayer: YieldClaim<
                    <Self::ClaimAggregator as ClaimAggregator<F>>::Claim,
                >,
            >,
        > + YieldClaim<<Self::ClaimAggregator as ClaimAggregator<F>>::Claim>
        + Serialize
        + for<'de> Deserialize<'de>
        + Debug;

    ///The logic that handles how to aggregate claims
    /// As this trait defines the 'bridge' between layers, some helper traits may be neccessary to implement
    /// on the other layer types
    type ClaimAggregator: ClaimAggregator<F, Layer = Self::Layer, InputLayer = Self::InputLayer>
        + Debug;
}

/// The default proof system for the remainder prover
#[derive(Clone, Debug, PartialEq)]
pub struct DefaultProofSystem;

impl<F: Field> ProofSystem<F> for DefaultProofSystem {
    type Layer = LayerEnum<F>;

    type InputLayer = InputLayerEnum<F>;

    type Transcript = PoseidonSponge<F>;

    type OutputLayer = MleOutputLayer<F>;

    type ClaimAggregator = WLXAggregator<F, Self::Layer, Self::InputLayer>;
}

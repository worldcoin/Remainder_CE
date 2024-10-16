///This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType), ..)
#[macro_export]
macro_rules! layer_enum {
    ($type_name:ident, $(($var_name:ident: $variant:ty)),+) => {

        paste::paste! {
            #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Remainder generated trait enum"]
            pub enum [<$type_name Enum>]<F: Field> {
                $(
                    #[doc = "Remainder generated layer variant"]
                    $var_name(Box<$variant>),
                )*
            }

            impl<F: Field> $crate::layer::LayerDescription<F> for [<$type_name DescriptionEnum>]<F> {
                type VerifierLayer = [<Verifier $type_name Enum>]<F>;

                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn compute_data_outputs(
                    &self,
                    mle_outputs_necessary: &std::collections::HashSet<&$crate::expression::circuit_expr::MleDescription<F>>,
                    circuit_map: &mut $crate::layouter::layouting::CircuitMap<F>,
                ) {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.compute_data_outputs(mle_outputs_necessary, circuit_map),
                        )*
                    }
                }

                fn verify_rounds(
                    &self,
                    claim: $crate::claims::Claim<F>,
                    transcript: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> Result<VerifierLayerEnum<F>, super::VerificationError> {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(layer.verify_rounds(claim, transcript)?),
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

                fn convert_into_verifier_layer(
                    &self,
                    sumcheck_bindings: &[F],
                    claim_point: &[F],
                    transcript_reader: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> Result<Self::VerifierLayer, super::VerificationError> {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(Self::VerifierLayer::$var_name(layer.convert_into_verifier_layer(sumcheck_bindings, claim_point, transcript_reader)?)),
                        )*
                    }
                }

                fn get_circuit_mles(
                    &self,
                ) -> Vec<& $crate::expression::circuit_expr::MleDescription<F>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_circuit_mles(),
                        )*
                    }
                }

                fn index_mle_indices(
                    &mut self, start_index: usize,
                ) {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.index_mle_indices(start_index),
                        )*
                    }
                }

                fn convert_into_prover_layer(
                    &self,
                    circuit_map: &$crate::layouter::layouting::CircuitMap<F>
                ) -> LayerEnum<F> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.convert_into_prover_layer(circuit_map),
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

            impl<F: Field> $crate::layer::VerifierLayer<F> for [<Verifier$type_name Enum>]<F> {
                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }
            }

            impl<F: Field> $crate::layer::Layer<F> for [<$type_name Enum>]<F> {
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

                fn compute_round_sumcheck_message(&mut self, round_index: usize) -> Result<Vec<F>, super::LayerError> {
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
                ) -> $crate::layer::PostSumcheckLayer<F, F> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_post_sumcheck_layer(round_challenges, claim_challenges),
                        )*
                    }
                }
            }

        $(
            impl<F: Field> From<$variant> for [<$type_name Enum>]<F> {
                fn from(var: $variant) -> [<$type_name Enum>]<F> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*
            impl<F: Field> YieldClaim<ClaimMle<F>> for [<Verifier $type_name Enum>]<F> {
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
            pub enum [<$type_name ProverCommitment>]<F: Field> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::ProverCommitment),
                )*
            }

            #[derive(serde::Serialize, serde::Deserialize, Debug)]
            #[serde(bound = "F: Field")]
            #[doc = r"Remainder generated commitment enum"]
            pub enum [<$type_name VerifierCommitment>]<F: Field> {
                $(
                    #[doc = "Remainder generated Commitment variant"]
                    $var_name(<$variant as InputLayer<F>>::VerifierCommitment),
                )*
            }
        }

        impl<F: Field> $crate::input_layer::InputLayer<F> for $type_name<F> {
            paste::paste! {
                type ProverCommitment = [<$type_name ProverCommitment>]<F>;
                type VerifierCommitment = [<$type_name VerifierCommitment>]<F>;
            }

            fn commit(&mut self) -> Result<Self::VerifierCommitment, $crate::input_layer::InputLayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => {
                            Ok(Self::VerifierCommitment::$var_name(layer.commit()?))
                        }
                    )*
                }
            }

            fn append_commitment_to_transcript(
                commitment: &Self::VerifierCommitment,
                transcript_writer: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<F>,
            ) {
                match commitment {
                    $(
                        Self::VerifierCommitment::$var_name(commitment) => <$variant as InputLayer<F>>::append_commitment_to_transcript(commitment, transcript_writer),
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
        // LigeroInputLayer::new()
        // InputLayerEnum::new()

        $(
            impl<F: Field> From<$variant> for $type_name<F> {
                fn from(var: $variant) -> $type_name<F> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*
    }
}

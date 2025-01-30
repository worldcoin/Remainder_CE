/// This macro generates a layer enum that represents all the possible layers
/// Every layer variant of the enum needs to implement Layer, and the enum will also implement Layer and pass methods to it's variants
///
/// Usage:
///
/// layer_enum!(EnumName, (FirstVariant: LayerType), (SecondVariant: SecondLayerType), ..)
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
                    mle_outputs_necessary: &std::collections::HashSet<&$crate::mle::mle_description::MleDescription<F>>,
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
                    claim: $crate::claims::RawClaim<F>,
                    transcript: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<F>,
                ) -> anyhow::Result<VerifierLayerEnum<F>> {
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
                ) -> anyhow::Result<Self::VerifierLayer> {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(Self::VerifierLayer::$var_name(layer.convert_into_verifier_layer(sumcheck_bindings, claim_point, transcript_reader)?)),
                        )*
                    }
                }

                fn get_circuit_mles(
                    &self,
                ) -> Vec<& $crate::mle::mle_description::MleDescription<F>> {
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

                fn get_claims(&self) -> anyhow::Result<Vec<$crate::claims::Claim<F>>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_claims(),
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

                fn prove(
                    &mut self,
                    claim: $crate::claims::RawClaim<F>,
                    transcript: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<F>,
                ) -> anyhow::Result<()> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.prove(claim, transcript),
                        )*
                    }
                }

                fn initialize(&mut self, claim_point: &[F]) -> anyhow::Result<()> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.initialize(claim_point),
                        )*
                    }
                }

                fn compute_round_sumcheck_message(&mut self, round_index: usize) -> anyhow::Result<Vec<F>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.compute_round_sumcheck_message(round_index),
                        )*
                    }
                }

                fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> anyhow::Result<()> {
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

                fn get_claims(&self) -> anyhow::Result<Vec<$crate::claims::Claim<F>>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_claims(),
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
        }
    }
}

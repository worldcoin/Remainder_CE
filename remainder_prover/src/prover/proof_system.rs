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
            #[serde(bound = "E: ExtensionField")]
            #[doc = r"Remainder generated trait enum"]
            pub enum [<$type_name Enum>]<E: ExtensionField> {
                $(
                    #[doc = "Remainder generated layer variant"]
                    $var_name(Box<$variant>),
                )*
            }

            impl<F: Field> $crate::layer::LayerDescription<F> for [<$type_name DescriptionEnum>]<F> {
                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn compute_data_outputs<E>(
                    &self,
                    mle_outputs_necessary: &std::collections::HashSet<&$crate::mle::mle_description::MleDescription>,
                    circuit_map: &mut $crate::circuit_layout::CircuitEvalMap<E>,
                )
                where
                    E: ExtensionField<BaseField = F>
                {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.compute_data_outputs(mle_outputs_necessary, circuit_map),
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

                fn get_circuit_mles(
                    &self,
                ) -> Vec<& $crate::mle::mle_description::MleDescription> {
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

                fn convert_into_prover_layer<E>(
                    &self,
                    circuit_map: &$crate::circuit_layout::CircuitEvalMap<E>
                ) -> LayerEnum<E>
                where
                    E: ExtensionField<BaseField = F>
                {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.convert_into_prover_layer(circuit_map),
                        )*
                    }
                }

                fn verify_rounds<E>(
                    &self,
                    claims: &[&$crate::claims::RawClaim<E>],
                    transcript: &mut impl $crate::remainder_shared_types::transcript::VerifierTranscript<E::BaseField>,
                ) -> anyhow::Result<VerifierLayerEnum<E>>
                where
                    E: ExtensionField<BaseField = F>,
                {
                    match self {
                        $(
                            Self::$var_name(layer) => Ok(layer.verify_rounds(claims, transcript)?),
                        )*
                    }
                }

                fn get_post_sumcheck_layer<E>(
                    &self,
                    round_challenges: &[E],
                    claim_challenges: &[&[E]],
                    random_coefficients: &[E],
                ) -> $crate::layer::PostSumcheckLayer<E, Option<E>>
                where
                    E: ExtensionField<BaseField = F>,
                {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_post_sumcheck_layer(round_challenges, claim_challenges, random_coefficients),
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

            impl<E: ExtensionField> $crate::layer::VerifierLayer<E> for [<Verifier$type_name Enum>]<E> {
                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn get_claims(&self) -> anyhow::Result<Vec<$crate::claims::Claim<E>>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_claims(),
                        )*
                    }
                }
            }

            impl<E: ExtensionField> $crate::layer::Layer<E> for [<$type_name Enum>]<E> {
                fn layer_id(&self) -> super::LayerId {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.layer_id(),
                        )*
                    }
                }

                fn prove(
                    &mut self,
                    claims: &[&$crate::claims::RawClaim<E>],
                    transcript: &mut impl $crate::remainder_shared_types::transcript::ProverTranscript<E::BaseField>,
                ) -> anyhow::Result<()> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.prove(claims, transcript),
                        )*
                    }
                }

                fn initialize(&mut self, claim_point: &[E]) -> anyhow::Result<()> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.initialize(claim_point),
                        )*
                    }
                }

                fn compute_round_sumcheck_message(&mut self, round_index: usize, random_coefficients: &[E]) -> anyhow::Result<Vec<E>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.compute_round_sumcheck_message(round_index, random_coefficients),
                        )*
                    }
                }

                fn bind_round_variable(&mut self, round_index: usize, challenge: E) -> anyhow::Result<()> {
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
                    round_challenges: &[E],
                    claim_challenges: &[&[E]],
                    random_coefficients: &[E],
                ) -> $crate::layer::PostSumcheckLayer<E, E> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_post_sumcheck_layer(round_challenges, claim_challenges, random_coefficients),
                        )*
                    }
                }

                fn get_claims(&self) -> anyhow::Result<Vec<$crate::claims::Claim<E>>> {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.get_claims(),
                        )*
                    }
                }

                fn initialize_rlc(&mut self, random_coefficients: &[E], claims: &[&$crate::claims::RawClaim<E>]) {
                    match self {
                        $(
                            Self::$var_name(layer) => layer.initialize_rlc(random_coefficients, claims),
                        )*
                    }
                }

            }

        $(
            impl<E: ExtensionField> From<$variant> for [<$type_name Enum>]<E> {
                fn from(var: $variant) -> [<$type_name Enum>]<E> {
                    Self::$var_name(Box::new(var))
                }
            }
        )*
        }
    }
}

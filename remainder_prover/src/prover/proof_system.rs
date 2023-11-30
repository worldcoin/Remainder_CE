use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::layer::{Layer, GKRLayer};

use super::input_layer::InputLayer;

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

        impl<F: FieldExt> $crate::layer::Layer<F> for $type_name<F> {
            fn prove_rounds(
                &mut self,
                claim: $crate::layer::claims::Claim<F>,
                transcript: &mut impl remainder_shared_types::transcript::Transcript<F>,
            ) -> Result<$crate::prover::SumcheckProof<F>, super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.prove_rounds(claim, transcript),
                    )*
                }
            }

            fn verify_rounds(
                &mut self,
                claim: $crate::layer::claims::Claim<F>,
                sumcheck_rounds: Vec<Vec<F>>,
                transcript: &mut impl remainder_shared_types::transcript::Transcript<F>,
            ) -> Result<(), super::LayerError> {
                match self {
                    $(
                        Self::$var_name(layer) => layer.verify_rounds(claim, sumcheck_rounds, transcript),
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
    }
}

///A trait for bundling a group of types that define the interfaces that go into a GKR Prover
pub trait ProofSystem<F: FieldExt> {
    ///A trait that defines the allowed Layer for this ProofSystem
    type Layer: Layer<F>;
    ///A trait that defines the allowed InputLayer for this ProofSystem
    type InputLayer: InputLayer<F>;

}
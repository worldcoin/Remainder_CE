use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::{
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    layer::{
        batched::{self, combine_zero_mle_ref, BatchedLayer},
        layer_enum::LayerEnum,
        simple_builders::ZeroBuilder,
        LayerBuilder, LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        Mle, MleIndex, MleRef,
    },
    prover::{
        combine_layers::combine_layers,
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder,
            enum_input_layer::{CommitmentEnum, InputLayerEnum},
            public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, GKRError, Layers, Witness,
    },
};
use remainder_shared_types::{
    transcript::{self, poseidon_transcript::PoseidonSponge, TranscriptWriter},
    FieldExt, Fr,
};
use tracing::instrument;
use utils::{
    ConstantScaledSumBuilder, ProductScaledBuilder, ProductSumBuilder, TripleNestedSelectorBuilder,
};

use crate::utils::{get_dummy_one_mle, get_dummy_random_mle};
mod utils;

struct ProductScaledSumCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ProductScaledSumCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builder = ProductScaledBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder = ProductSumBuilder::new(first_layer_output, self.mle_1.clone());
        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builder = ZeroBuilder::new(second_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct SumConstantCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for SumConstantCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builder = ProductSumBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder =
            ConstantScaledSumBuilder::new(first_layer_output, self.mle_1.clone());
        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builder = ZeroBuilder::new(second_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct ConstantScaledCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ConstantScaledCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builder =
            ConstantScaledSumBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder =
            ProductScaledBuilder::new(first_layer_output, self.mle_1.clone());
        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builder = ZeroBuilder::new(second_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

struct CombinedCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle_1), Box::new(&mut self.mle_2)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        let mut pss_circuit = ProductScaledSumCircuit {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };
        let mut sc_circuit = SumConstantCircuit {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };
        let mut cs_circuit = ConstantScaledCircuit {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };

        let pss_witness = pss_circuit.synthesize();
        let sc_witness = sc_circuit.synthesize();
        let cs_witness = cs_circuit.synthesize();

        let Witness {
            layers: pss_layers,
            output_layers: pss_output_layers,
            input_layers: _,
        } = pss_witness;

        let Witness {
            layers: sc_layers,
            output_layers: sc_output_layers,
            input_layers: _,
        } = sc_witness;

        let Witness {
            layers: cs_layers,
            output_layers: cs_output_layers,
            input_layers: _,
        } = cs_witness;

        let (layers, output_layers) = combine_layers(
            vec![pss_layers, sc_layers, cs_layers],
            vec![pss_output_layers, sc_output_layers, cs_output_layers],
        )
        .unwrap();

        Witness {
            layers,
            output_layers: output_layers,
            input_layers: vec![input_layer],
        }
    }
}

impl<F: FieldExt> CombinedCircuit<F> {
    fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>) -> Self {
        assert_eq!(mle_1.num_iterated_vars(), mle_2.num_iterated_vars());
        Self { mle_1, mle_2 }
    }
}

#[test]
fn test_combined_circuit() {
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let mle_1 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);
    let mle_2 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);

    let combined_circuit: CombinedCircuit<Fr> = CombinedCircuit::new(mle_1, mle_2);
    test_circuit(combined_circuit, None)
}

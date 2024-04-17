use ark_std::test_rng;

use remainder::{
    layer::{layer_builder::simple_builders::ZeroBuilder, LayerId},
    mle::{dense::DenseMle, Mle, MleRef},
    prover::{
        combine_layers::combine_layers,
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
        },
        proof_system::DefaultProofSystem,
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{FieldExt, Fr};

use utils::{ConstantScaledSumBuilder, ProductScaledBuilder, ProductSumBuilder};

use crate::utils::get_dummy_random_mle;
mod utils;

/// A circuit which takes in two MLEs of the same size:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ProductSumBuilder] with the output of Layer 0 and `mle_1`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1`  An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
struct ProductScaledSumCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ProductScaledSumCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
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

/// A circuit which takes in two MLEs of the same size:
/// * Layer 0: [ProductSumBuilder] with the two inputs
/// * Layer 1: [ConstantScaledSumBuilder] with the output of Layer 0 and `mle_1`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
struct SumConstantCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for SumConstantCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
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

/// A circuit which takes in two MLEs of the same size:
/// * Layer 0: [ConstantScaledSumBuilder] with the two inputs
/// * Layer 1: [ProductScaledBuilder] with the output of Layer 0 and `mle_1`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
struct ConstantScaledCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ConstantScaledCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
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

/// A circuit which combines the [ProductScaledSumCircuit], [SumConstantCircuit],
/// and [ConstantScaledCircuit].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1`.
struct CombinedCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle_1), Box::new(&mut self.mle_2)];
        let input_layer = InputLayerBuilder::new(input_mles, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

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

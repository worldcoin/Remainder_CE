use ark_std::test_rng;

use remainder::{
    builders::{
        combine_input_layers::InputLayerBuilder, combine_layers::combine_layers,
        layer_builder::simple_builders::ZeroBuilder,
    },
    input_layer::public_input_layer::PublicInputLayer,
    layer::LayerId,
    mle::{dense::DenseMle, Mle},
    prover::{
        helpers::test_circuit, layers::Layers, proof_system::DefaultProofSystem, GKRCircuit,
        Witness,
    },
};
use remainder_shared_types::{FieldExt, Fr};
use utils::{ConstantScaledSumBuilder, ProductScaledBuilder, ProductSumBuilder};

use crate::utils::get_dummy_random_mle_vec;
pub mod utils;

/// A circuit which takes in two vectors of MLEs of the same size:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ProductSumBuilder] with the output of Layer 0 and `mle_1_vec`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
struct DataParallelProductScaledSumCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelProductScaledSumCircuitAlt<F> {
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

/// A circuit which takes in two vectors of MLEs of the same size:
/// * Layer 0: [ProductSumBuilder] with the two inputs
/// * Layer 1: [ConstantScaledSumBuilder] with the output of Layer 0 and `mle_1_vec`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
struct DataParallelSumConstantCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelSumConstantCircuitAlt<F> {
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

/// A circuit which takes in two vectors of MLEs of the same size:
/// * Layer 0: [ConstantScaledSumBuilder] with the two inputs
/// * Layer 1: [ProductScaledBuilder] with the output of Layer 0 and `mle_1_vec`
/// * Layer 2: [ZeroBuilder] with output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
struct DataParallelConstantScaledCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelConstantScaledCircuitAlt<F> {
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

/// A circuit which combines the [DataParallelProductScaledSumCircuit], [DataParallelSumConstantCircuit],
/// and [DataParallelConstantScaledCircuit].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct DataParallelCombinedCircuitAlt<F: FieldExt> {
    mle_1: DenseMle<F>,
    mle_2: DenseMle<F>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCombinedCircuitAlt<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        self.mle_1.layer_id = LayerId::Input(0);
        self.mle_2.layer_id = LayerId::Input(0);
        let input_mles: Vec<&mut dyn Mle<F>> = vec![&mut self.mle_1, &mut self.mle_2];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut pss_circuit = DataParallelProductScaledSumCircuitAlt {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };
        let mut sc_circuit = DataParallelSumConstantCircuitAlt {
            mle_1: self.mle_1.clone(),
            mle_2: self.mle_2.clone(),
        };
        let mut cs_circuit = DataParallelConstantScaledCircuitAlt {
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

impl<F: FieldExt> DataParallelCombinedCircuitAlt<F> {
    fn new(mle_1: DenseMle<F>, mle_2: DenseMle<F>) -> Self {
        Self { mle_1, mle_2 }
    }
}

#[test]
fn test_combined_dataparallel_circuit_alt() {
    const NUM_DATAPARALLEL_BITS: usize = 1;
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let mle_1_vec = get_dummy_random_mle_vec(VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);

    // These checks can possibly be done with the newly designed batching bits/system
    let all_num_vars: Vec<usize> = mle_1_vec
        .iter()
        .chain(mle_2_vec.iter())
        .map(|mle| mle.num_iterated_vars())
        .collect();
    let all_vars_same = all_num_vars.iter().fold(true, |acc, elem| {
        (*elem == mle_1_vec[0].num_iterated_vars()) & acc
    });
    assert!(all_vars_same);
    assert_eq!(mle_1_vec.len(), mle_2_vec.len());
    assert_eq!(mle_1_vec.len(), 1 << NUM_DATAPARALLEL_BITS);
    // These checks can possibly be done with the newly designed batching bits/system

    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits
    let mle_1_vec_batched = DenseMle::batch_mles(mle_1_vec);
    let mle_2_vec_batched = DenseMle::batch_mles(mle_2_vec);
    // the batched mle should be able to demonstrate that there's NUM_DATA_PARALLEL_BITS of batch bits

    let combined_circuit: DataParallelCombinedCircuitAlt<Fr> =
        DataParallelCombinedCircuitAlt::new(mle_1_vec_batched, mle_2_vec_batched);
    test_circuit(combined_circuit, None)
}

use ark_std::test_rng;
use itertools::{repeat_n, Itertools};
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

use crate::utils::{get_dummy_one_mle, get_dummy_random_mle, get_dummy_random_mle_vec};
mod utils;

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
struct DataParallelProductScaledSumCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelProductScaledSumCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builders = self
            .mle_1_vec
            .iter()
            .zip(self.mle_2_vec.iter())
            .map(|(mle_1, mle_2)| ProductScaledBuilder::new(mle_1.clone(), mle_2.clone()))
            .collect_vec();
        let first_layer_outputs = layers.add_gkr(BatchedLayer::new(first_layer_builders));

        let second_layer_builders = first_layer_outputs
            .into_iter()
            .zip(self.mle_1_vec.iter())
            .map(|(first_layer_output, mle_1)| {
                ProductSumBuilder::new(first_layer_output, mle_1.clone())
            })
            .collect_vec();
        let second_layer_outputs = layers.add_gkr(BatchedLayer::new(second_layer_builders));

        let zero_builders = second_layer_outputs
            .into_iter()
            .map(|second_layer_output| ZeroBuilder::new(second_layer_output))
            .collect_vec();
        let outputs = layers.add_gkr(BatchedLayer::new(zero_builders));
        let output = combine_zero_mle_ref(outputs);

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
struct DataParallelSumConstantCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelSumConstantCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builders = self
            .mle_1_vec
            .iter()
            .zip(self.mle_2_vec.iter())
            .map(|(mle_1, mle_2)| ProductSumBuilder::new(mle_1.clone(), mle_2.clone()))
            .collect_vec();
        let first_layer_outputs = layers.add_gkr(BatchedLayer::new(first_layer_builders));

        let second_layer_builders = first_layer_outputs
            .into_iter()
            .zip(self.mle_1_vec.iter())
            .map(|(first_layer_output, mle_1)| {
                ConstantScaledSumBuilder::new(first_layer_output, mle_1.clone())
            })
            .collect_vec();

        let second_layer_outputs = layers.add_gkr(BatchedLayer::new(second_layer_builders));

        let zero_builders = second_layer_outputs
            .into_iter()
            .map(|second_layer_output| ZeroBuilder::new(second_layer_output))
            .collect_vec();
        let outputs = layers.add_gkr(BatchedLayer::new(zero_builders));
        let output = combine_zero_mle_ref(outputs);

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
struct DataParallelConstantScaledCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelConstantScaledCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let first_layer_builders = self
            .mle_1_vec
            .iter()
            .zip(self.mle_2_vec.iter())
            .map(|(mle_1, mle_2)| ConstantScaledSumBuilder::new(mle_1.clone(), mle_2.clone()))
            .collect_vec();

        let first_layer_outputs = layers.add_gkr(BatchedLayer::new(first_layer_builders));

        let second_layer_builders = first_layer_outputs
            .into_iter()
            .zip(self.mle_1_vec.iter())
            .map(|(first_layer_output, mle_1)| {
                ProductScaledBuilder::new(first_layer_output, mle_1.clone())
            })
            .collect_vec();

        let second_layer_outputs = layers.add_gkr(BatchedLayer::new(second_layer_builders));

        let zero_builders = second_layer_outputs
            .into_iter()
            .map(|second_layer_output| ZeroBuilder::new(second_layer_output))
            .collect_vec();
        let outputs = layers.add_gkr(BatchedLayer::new(zero_builders));
        let output = combine_zero_mle_ref(outputs);

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
struct DataParallelCombinedCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    num_dataparallel_bits: usize,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCombinedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut combined_mle_1 = DenseMle::<F, F>::combine_mle_batch(self.mle_1_vec.clone());
        let mut combined_mle_2 = DenseMle::<F, F>::combine_mle_batch(self.mle_2_vec.clone());
        combined_mle_1.layer_id = LayerId::Input(0);
        combined_mle_2.layer_id = LayerId::Input(0);
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut combined_mle_1), Box::new(&mut combined_mle_2)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        self.mle_1_vec
            .iter_mut()
            .zip(self.mle_2_vec.iter_mut())
            .for_each(|(mle_1, mle_2)| {
                mle_1.set_prefix_bits(Some(
                    combined_mle_1
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(repeat_n(MleIndex::Iterated, self.num_dataparallel_bits))
                        .collect_vec(),
                ));
                mle_2.set_prefix_bits(Some(
                    combined_mle_2
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(repeat_n(MleIndex::Iterated, self.num_dataparallel_bits))
                        .collect_vec(),
                ));
            });

        let mut pss_circuit = DataParallelProductScaledSumCircuit {
            mle_1_vec: self.mle_1_vec.clone(),
            mle_2_vec: self.mle_2_vec.clone(),
        };
        let mut sc_circuit = DataParallelSumConstantCircuit {
            mle_1_vec: self.mle_1_vec.clone(),
            mle_2_vec: self.mle_2_vec.clone(),
        };
        let mut cs_circuit = DataParallelConstantScaledCircuit {
            mle_1_vec: self.mle_1_vec.clone(),
            mle_2_vec: self.mle_2_vec.clone(),
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

impl<F: FieldExt> DataParallelCombinedCircuit<F> {
    fn new(
        mle_1_vec: Vec<DenseMle<F, F>>,
        mle_2_vec: Vec<DenseMle<F, F>>,
        num_dataparallel_bits: usize,
    ) -> Self {
        let all_num_vars: Vec<usize> = mle_1_vec
            .iter()
            .chain(mle_2_vec.iter())
            .map(|mle| mle.num_iterated_vars())
            .collect();
        let all_vars_same = all_num_vars.iter().fold(true, |acc, elem| {
            (elem == all_num_vars.first().unwrap()) & acc
        });
        assert!(all_vars_same);
        assert_eq!(mle_1_vec.len(), mle_2_vec.len());
        assert_eq!(mle_1_vec.len(), 1 << num_dataparallel_bits);
        Self {
            mle_1_vec,
            mle_2_vec,
            num_dataparallel_bits,
        }
    }
}

#[test]
fn test_combined_circuit() {
    const NUM_DATAPARALLEL_BITS: usize = 1;
    const VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let mle_1_vec = get_dummy_random_mle_vec(VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(VARS_MLE_1_2, NUM_DATAPARALLEL_BITS, &mut rng);

    let combined_circuit: DataParallelCombinedCircuit<Fr> =
        DataParallelCombinedCircuit::new(mle_1_vec, mle_2_vec, NUM_DATAPARALLEL_BITS);
    test_circuit(combined_circuit, None)
}

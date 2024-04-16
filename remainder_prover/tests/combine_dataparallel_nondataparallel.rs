use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::{
        generic_expr::{Expression, ExpressionNode, ExpressionType},
        prover_expr::ProverExpr,
    },
    layer::{
        layer_builder::{
            batched::{combine_zero_mle_ref, BatchedLayer},
            simple_builders::ZeroBuilder,
        },
        layer_enum::LayerEnum,
        LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        Mle, MleIndex, MleRef,
    },
    prover::{
        combine_layers::combine_layers,
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, enum_input_layer::InputLayerEnum,
            public_input_layer::PublicInputLayer, InputLayer,
        },
        proof_system::DefaultProofSystem,
        GKRCircuit, GKRError, Layers, Witness,
    },
};
use remainder_shared_types::{
    transcript::{self, poseidon_transcript::PoseidonSponge, TranscriptWriter},
    FieldExt, Fr,
};
use tracing::instrument;
use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};

use crate::utils::get_dummy_random_mle;
mod utils;

/// A circuit which takes in two vectors of MLEs of the same size:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ProductScaledBuilder] with the output of Layer 0 and output of Layer 0.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec` - An MLE vec with arbitrary bookkeeping table values.
/// * `mle_2_vec` - An MLE vec with arbitrary bookkeeping table values, same size as `mle_1_vec`.
struct DataParallelCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builders = (self
            .mle_1_vec
            .clone()
            .into_iter()
            .zip(self.mle_2_vec.clone().into_iter()))
        .map(|(mle_1, mle_2)| ProductScaledBuilder::new(mle_1, mle_2))
        .collect_vec();
        let batched_first_layer_builder = BatchedLayer::new(first_layer_builders);
        let first_layer_output = layers.add_gkr(batched_first_layer_builder);

        let second_layer_builders = (first_layer_output.iter().zip(first_layer_output.clone()))
            .map(|(mle_1, mle_2)| ProductScaledBuilder::new(mle_1.clone(), mle_2))
            .collect_vec();

        let batched_second_layer_builder = BatchedLayer::new(second_layer_builders);
        let second_layer_output = layers.add_gkr(batched_second_layer_builder);

        let zero_builders = second_layer_output
            .iter()
            .map(|mle| ZeroBuilder::new(mle.clone()))
            .collect_vec();
        let batched_output_builder = BatchedLayer::new(zero_builders);
        let output_mles = layers.add_gkr(batched_output_builder);
        let output = combine_zero_mle_ref(output_mles);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

/// A circuit in which:
/// * Layer 0: [TripleNestedSelectorBuilder] with the three inputs
/// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_inner_sel_mle`
/// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
/// the size of `inner_sel_mle`
struct TripleNestedSelectorCircuit<F: FieldExt> {
    inner_inner_sel_mle: DenseMle<F, F>,
    inner_sel_mle: DenseMle<F, F>,
    outer_sel_mle: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for TripleNestedSelectorCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builder = TripleNestedSelectorBuilder::new(
            self.inner_inner_sel_mle.clone(),
            self.inner_sel_mle.clone(),
            self.outer_sel_mle.clone(),
        );
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

/// A circuit in which:
/// * Layer 0: [ProductScaledBuilder] with the two inputs
/// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values, same size as `mle_1`.
struct ScaledProductCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
}

impl<F: FieldExt> GKRCircuit<F> for ScaledProductCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let first_layer_builder = ProductScaledBuilder::new(self.mle_1.clone(), self.mle_2.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![],
        }
    }
}

/// A circuit which combines the [DataParallelCircuit], [TripleNestedSelectorCircuit],
/// and [ScaledProductCircuit].
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [DataParallelCircuit] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_4`, `mle_5`, `mle_6` - inputs to [TripleNestedSelectorCircuit], `mle_4` has the same
/// size as the mles in `mle_1_vec`, arbitrary bookkeeping table values. `mle_5` has one more
/// variable than `mle_4`, `mle_6` has one more variable than `mle_5`, both arbitrary bookkeeping
/// table values.
/// * `mle_3`, `mle_4` - inputs to [ScaledProductCircuit], both arbitrary bookkeeping table values,
/// same size.
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct CombinedCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    mle_3: DenseMle<F, F>,
    mle_4: DenseMle<F, F>,
    mle_5: DenseMle<F, F>,
    mle_6: DenseMle<F, F>,
    num_data_parallel_bits: usize,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut mle_1_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_1_vec.clone());
        let mut mle_2_combined = DenseMle::<F, F>::combine_mle_batch(self.mle_2_vec.clone());
        mle_1_combined.layer_id = LayerId::Input(0);
        mle_2_combined.layer_id = LayerId::Input(0);
        self.mle_1_vec
            .iter_mut()
            .zip(self.mle_2_vec.iter_mut())
            .for_each(|(mle_1, mle_2)| {
                mle_1.layer_id = LayerId::Input(0);
                mle_2.layer_id = LayerId::Input(0);
            });
        self.mle_3.layer_id = LayerId::Input(0);
        self.mle_4.layer_id = LayerId::Input(0);
        self.mle_5.layer_id = LayerId::Input(0);
        self.mle_6.layer_id = LayerId::Input(0);

        let input_commit: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut mle_1_combined),
            Box::new(&mut mle_2_combined),
            Box::new(&mut self.mle_3),
            Box::new(&mut self.mle_4),
            Box::new(&mut self.mle_5),
            Box::new(&mut self.mle_6),
        ];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F>>();

        let mut input_layer_enum = input_layer.into();

        self.mle_1_vec
            .iter_mut()
            .zip(self.mle_2_vec.iter_mut())
            .for_each(|(mle_1, mle_2)| {
                mle_1.set_prefix_bits(Some(
                    mle_1_combined
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_data_parallel_bits])
                        .collect_vec(),
                ));
                mle_2.set_prefix_bits(Some(
                    mle_2_combined
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_data_parallel_bits])
                        .collect_vec(),
                ));
            });

        let mut batched_circuit = DataParallelCircuit {
            mle_1_vec: self.mle_1_vec.clone(),
            mle_2_vec: self.mle_2_vec.clone(),
        };
        let mut triple_nested_sel_circuit = TripleNestedSelectorCircuit {
            inner_inner_sel_mle: self.mle_4.clone(),
            inner_sel_mle: self.mle_5.clone(),
            outer_sel_mle: self.mle_6.clone(),
        };
        let mut product_scaled_circuit = ScaledProductCircuit {
            mle_1: self.mle_3.clone(),
            mle_2: self.mle_4.clone(),
        };

        let batched_circuit_witness = batched_circuit.synthesize();
        let triple_nested_sel_witness = triple_nested_sel_circuit.synthesize();
        let product_scaled_witness = product_scaled_circuit.synthesize();

        let Witness {
            layers: batched_layers,
            output_layers: batched_output_layers,
            input_layers: _,
        } = batched_circuit_witness;

        let Witness {
            layers: triple_nested_layers,
            output_layers: triple_output_layers,
            input_layers: _,
        } = triple_nested_sel_witness;

        let Witness {
            layers: product_scaled_layers,
            output_layers: product_scaled_output_layers,
            input_layers: _,
        } = product_scaled_witness;

        let (layers, output_layers) = combine_layers(
            vec![batched_layers, triple_nested_layers, product_scaled_layers],
            vec![
                batched_output_layers,
                triple_output_layers,
                product_scaled_output_layers,
            ],
        )
        .unwrap();

        Witness {
            layers,
            output_layers,
            input_layers: vec![input_layer_enum],
        }
    }
}

impl<F: FieldExt> CombinedCircuit<F> {
    fn new(
        mle_1_vec: Vec<DenseMle<F, F>>,
        mle_2_vec: Vec<DenseMle<F, F>>,
        mle_3: DenseMle<F, F>,
        mle_4: DenseMle<F, F>,
        mle_5: DenseMle<F, F>,
        mle_6: DenseMle<F, F>,
        num_data_parallel_bits: usize,
    ) -> Self {
        assert_eq!(mle_3.num_iterated_vars(), mle_4.num_iterated_vars());
        assert_eq!(mle_5.num_iterated_vars(), mle_4.num_iterated_vars() + 1);
        assert_eq!(mle_6.num_iterated_vars(), mle_5.num_iterated_vars() + 1);
        let all_num_vars: Vec<usize> = mle_1_vec
            .iter()
            .chain(mle_2_vec.iter())
            .map(|mle| mle.num_iterated_vars())
            .collect();
        let all_vars_same = all_num_vars
            .iter()
            .fold(true, |acc, elem| (*elem == mle_3.num_iterated_vars()) & acc);
        assert!(all_vars_same);
        assert_eq!(mle_1_vec.len(), mle_2_vec.len());
        assert_eq!(mle_1_vec.len(), 1 << num_data_parallel_bits);
        Self {
            mle_1_vec,
            mle_2_vec,
            mle_3,
            mle_4,
            mle_5,
            mle_6,
            num_data_parallel_bits,
        }
    }
}

#[test]
fn test_combined_circuit() {
    const VARS_MLE_1_2: usize = 2;
    const VARS_MLE_3: usize = VARS_MLE_1_2 + 1;
    const VARS_MLE_4: usize = VARS_MLE_3 + 1;
    const NUM_DATA_PARALLEL_BITS: usize = 1;
    let mut rng = test_rng();

    let mle_1_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_2_vec = (0..1 << NUM_DATA_PARALLEL_BITS)
        .map(|_| get_dummy_random_mle(VARS_MLE_1_2, &mut rng))
        .collect_vec();
    let mle_3 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);
    let mle_4 = get_dummy_random_mle(VARS_MLE_1_2, &mut rng);
    let mle_5 = get_dummy_random_mle(VARS_MLE_3, &mut rng);
    let mle_6 = get_dummy_random_mle(VARS_MLE_4, &mut rng);

    let combined_circuit: CombinedCircuit<Fr> = CombinedCircuit::new(
        mle_1_vec,
        mle_2_vec,
        mle_3,
        mle_4,
        mle_5,
        mle_6,
        NUM_DATA_PARALLEL_BITS,
    );
    test_circuit(combined_circuit, None)
}

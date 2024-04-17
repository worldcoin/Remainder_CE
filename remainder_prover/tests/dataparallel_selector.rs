use itertools::Itertools;

use remainder::{
    layer::{
        layer_builder::{
            batched::{combine_zero_mle_ref, BatchedLayer},
            simple_builders::ZeroBuilder,
        },
        LayerId,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
        },
        proof_system::DefaultProofSystem,
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::FieldExt;
use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};

pub mod utils;

/// A circuit which does the following:
/// * Layer 0: [ProductScaledBuilder] with `mle_1_vec`, `mle_2_vec`
/// * Layer 1: [TripleNestedSelectorBuilder] with output of Layer 0, `mle_3_vec`, `mle_4_vec`
/// * Layer 2: [ZeroBuilder] with the output of Layer 1 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_vec`, `mle_2_vec` - inputs to [ProductScaledBuilder] both arbitrary bookkeeping
/// table values, same size.
/// * `mle_3_vec`, `mle_4_vec` - inputs to [TripleNestedSelectorBuilder], both arbitrary bookkeeping table values,
/// `mle_3_vec` mles have one more variable than in `mle_1_vec`, `mle_2_vec`, and `mle_4_vec` mles
/// have one more variable than in `mle_3_vec`.
///
/// * `num_dataparallel_bits` - The number of bits that represent which copy index the circuit is.
struct DataParallelCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    mle_3_vec: Vec<DenseMle<F, F>>,
    mle_4_vec: Vec<DenseMle<F, F>>,
    num_dataparallel_bits: usize,
}

impl<F: FieldExt> GKRCircuit<F> for DataParallelCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let mut layers = Layers::new();

        let mut combined_mle_1 = DenseMle::<F, F>::combine_mle_batch(self.mle_1_vec.clone());
        let mut combined_mle_2 = DenseMle::<F, F>::combine_mle_batch(self.mle_2_vec.clone());
        let mut combined_mle_3 = DenseMle::<F, F>::combine_mle_batch(self.mle_3_vec.clone());
        let mut combined_mle_4 = DenseMle::<F, F>::combine_mle_batch(self.mle_4_vec.clone());
        combined_mle_1.layer_id = LayerId::Input(0);
        combined_mle_2.layer_id = LayerId::Input(0);
        combined_mle_3.layer_id = LayerId::Input(0);
        combined_mle_4.layer_id = LayerId::Input(0);

        let input_commit: Vec<&mut dyn Mle<F>> = vec![
            &mut combined_mle_1,
            &mut combined_mle_2,
            &mut combined_mle_3,
            &mut combined_mle_4,
        ];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F>>();

        let input_layer_enum = input_layer.into();

        self.mle_1_vec
            .iter_mut()
            .zip(
                self.mle_2_vec
                    .iter_mut()
                    .zip(self.mle_3_vec.iter_mut().zip(self.mle_4_vec.iter_mut())),
            )
            .for_each(|(mle_1, (mle_2, (mle_3, mle_4)))| {
                mle_1.layer_id = LayerId::Input(0);
                mle_2.layer_id = LayerId::Input(0);
                mle_3.layer_id = LayerId::Input(0);
                mle_4.layer_id = LayerId::Input(0);
                mle_1.set_prefix_bits(Some(
                    combined_mle_1
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_dataparallel_bits])
                        .collect_vec(),
                ));
                mle_2.set_prefix_bits(Some(
                    combined_mle_2
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_dataparallel_bits])
                        .collect_vec(),
                ));
                mle_3.set_prefix_bits(Some(
                    combined_mle_3
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_dataparallel_bits])
                        .collect_vec(),
                ));
                mle_4.set_prefix_bits(Some(
                    combined_mle_4
                        .get_prefix_bits()
                        .unwrap()
                        .into_iter()
                        .chain(vec![MleIndex::Iterated; self.num_dataparallel_bits])
                        .collect_vec(),
                ));
            });

        let first_layer_builders = (self
            .mle_1_vec
            .clone()
            .into_iter()
            .zip(self.mle_2_vec.clone().into_iter()))
        .map(|(mle_1, mle_2)| ProductScaledBuilder::new(mle_1, mle_2))
        .collect_vec();
        let batched_first_layer_builder = BatchedLayer::new(first_layer_builders);
        let first_layer_output = layers.add_gkr(batched_first_layer_builder);

        let second_layer_builders = (first_layer_output
            .iter()
            .zip(self.mle_3_vec.clone().iter().zip(self.mle_4_vec.clone())))
        .map(|(mle_1, (mle_2, mle_3))| {
            TripleNestedSelectorBuilder::new(mle_1.clone(), mle_2.clone(), mle_3.clone())
        })
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
            input_layers: vec![input_layer_enum],
        }
    }
}

impl<F: FieldExt> DataParallelCircuit<F> {
    fn _new(
        mle_1_vec: Vec<DenseMle<F, F>>,
        mle_2_vec: Vec<DenseMle<F, F>>,
        mle_3_vec: Vec<DenseMle<F, F>>,
        mle_4_vec: Vec<DenseMle<F, F>>,
        num_dataparallel_bits: usize,
    ) -> Self {
        assert_eq!(mle_1_vec.len(), mle_2_vec.len());
        assert_eq!(mle_3_vec.len(), mle_2_vec.len());
        assert_eq!(mle_3_vec.len(), mle_4_vec.len());
        assert_eq!(mle_1_vec.len(), 1 << num_dataparallel_bits);
        let all_num_vars_1_2: Vec<usize> = mle_1_vec
            .iter()
            .chain(mle_2_vec.iter())
            .map(|mle| mle.num_iterated_vars())
            .collect();
        let all_vars_same_1_2 = all_num_vars_1_2.iter().fold(true, |acc, elem| {
            (*elem == mle_3_vec[0].num_iterated_vars() - 1) & acc
        });
        assert!(all_vars_same_1_2);
        let all_num_vars_3: Vec<usize> = mle_3_vec
            .iter()
            .map(|mle| mle.num_iterated_vars())
            .collect();
        let all_vars_same_3 = all_num_vars_3.iter().fold(true, |acc, elem| {
            (*elem == mle_4_vec[0].num_iterated_vars() - 1) & acc
        });
        assert!(all_vars_same_3);
        let all_num_vars_4: Vec<usize> = mle_4_vec
            .iter()
            .map(|mle| mle.num_iterated_vars())
            .collect();
        let all_vars_same_4 = all_num_vars_4.iter().fold(true, |acc, elem| {
            (*elem == mle_4_vec[0].num_iterated_vars()) & acc
        });
        assert!(all_vars_same_4);
        Self {
            mle_1_vec,
            mle_2_vec,
            mle_3_vec,
            mle_4_vec,
            num_dataparallel_bits,
        }
    }
}

// TODO(vishady): this test fails based off of our current implementation of remainder!! The current problem is the way
// selector bits are treated when combining expressions.

// #[test]
// fn test_dataparallel_selector() {
//     const NUM_DATA_PARALLEL_BITS: usize = 3;
//     const NUM_VARS_MLE_1_2: usize = 2;
//     const NUM_VARS_MLE_3: usize = NUM_VARS_MLE_1_2 + 1;
//     const NUM_VARS_MLE_4: usize = NUM_VARS_MLE_3 + 1;
//     let mut rng = test_rng();

//     let mle_1_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
//     let mle_2_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
//     let mle_3_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_3, NUM_DATA_PARALLEL_BITS, &mut rng);
//     let mle_4_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_4, NUM_DATA_PARALLEL_BITS, &mut rng);

//     let circuit = DataParallelCircuit::new(
//         mle_1_vec,
//         mle_2_vec,
//         mle_3_vec,
//         mle_4_vec,
//         NUM_DATA_PARALLEL_BITS,
//     );
//     test_circuit(circuit, None);
// }

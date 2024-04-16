use std::cmp::max;

use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::{
    expression::generic_expr::Expression,
    layer::{
        batched::{combine_mles, combine_zero_mle_ref, BatchedLayer},
        from_mle,
        simple_builders::ZeroBuilder,
        LayerId,
    },
    mle::{
        dense::{DenseMle, Tuple2},
        zero::ZeroMleRef,
        Mle, MleIndex, MleRef,
    },
    prover::{
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
    transcript::{poseidon_transcript::PoseidonSponge, TranscriptWriter},
    FieldExt, Fr,
};
use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};

use crate::utils::get_dummy_random_mle_vec;
mod utils;
struct NonSelectorDataparallelCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    num_dataparallel_bits: usize,
}

impl<F: FieldExt> GKRCircuit<F> for NonSelectorDataparallelCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let mut layers = Layers::new();

        let mut combined_mle_1 = DenseMle::<F, F>::combine_mle_batch(self.mle_1_vec.clone());
        let mut combined_mle_2 = DenseMle::<F, F>::combine_mle_batch(self.mle_2_vec.clone());
        combined_mle_1.layer_id = LayerId::Input(0);
        combined_mle_2.layer_id = LayerId::Input(0);

        let input_commit: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut combined_mle_1), Box::new(&mut combined_mle_2)];

        let input_commit_builder =
            InputLayerBuilder::<F>::new(input_commit, None, LayerId::Input(0));

        let input_layer: PublicInputLayer<F, Self::Sponge> =
            input_commit_builder.to_input_layer::<PublicInputLayer<F, Self::Sponge>>();

        let input_layer_enum = input_layer.to_enum();

        self.mle_1_vec
            .iter_mut()
            .zip(self.mle_2_vec.iter_mut())
            .for_each(|(mle_1, mle_2)| {
                mle_1.layer_id = LayerId::Input(0);
                mle_2.layer_id = LayerId::Input(0);
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

        let zero_builders = first_layer_output
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

impl<F: FieldExt> NonSelectorDataparallelCircuit<F> {
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
            (*elem == mle_1_vec[0].num_iterated_vars()) & acc
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
fn test_gkr_simplest_batched_circuit() {
    const NUM_DATA_PARALLEL_BITS: usize = 3;
    const NUM_VARS_MLE_1_2: usize = 2;
    let mut rng = test_rng();

    let mle_1_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);
    let mle_2_vec = get_dummy_random_mle_vec(NUM_VARS_MLE_1_2, NUM_DATA_PARALLEL_BITS, &mut rng);

    let circuit = NonSelectorDataparallelCircuit::new(mle_1_vec, mle_2_vec, NUM_DATA_PARALLEL_BITS);
    test_circuit(circuit, None);
}

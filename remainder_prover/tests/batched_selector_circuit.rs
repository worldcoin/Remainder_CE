use std::cmp::max;

use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::{
    expression::generic_expr::Expression,
    layer::{
        batched::{combine_zero_mle_ref, BatchedLayer},
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
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt, Fr};
use utils::{ProductScaledBuilder, TripleNestedSelectorBuilder};
mod utils;
struct BatchedCircuit<F: FieldExt> {
    mle_1_vec: Vec<DenseMle<F, F>>,
    mle_2_vec: Vec<DenseMle<F, F>>,
    mle_3_vec: Vec<DenseMle<F, F>>,
    mle_4_vec: Vec<DenseMle<F, F>>,
}

impl<F: FieldExt> GKRCircuit<F> for BatchedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
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
            input_layers: vec![],
        }
    }
}

#[test]
fn test_gkr_simplest_batched_circuit() {
    let mut rng = test_rng();
    let size = 1 << 3;

    let batch_size = 1 << 2;
    // --- This should be 2^2 ---
    let batched_mle: Vec<DenseMle<Fr, Tuple2<Fr>>> = (0..batch_size)
        .map(|_idx1| {
            DenseMle::new_from_iter(
                (0..size).map(|_idx| {
                    let num = Fr::from(rng.gen::<u64>());
                    //let second_num = Fr::from(rng.gen::<u64>());
                    // let num = Fr::from(idx + idx1);
                    (num, num).into()
                }),
                LayerId::Input(0),
                None,
            )
        })
        .collect_vec();
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input(0),
    //     None,
    // );

    let circuit: SimplestBatchedCircuit<Fr> = SimplestBatchedCircuit {
        batched_first_second_mle: batched_mle,
        batch_bits: 2,
    };
    test_circuit(circuit, None);
}

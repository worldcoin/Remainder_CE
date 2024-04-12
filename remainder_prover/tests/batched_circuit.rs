use std::cmp::max;

use ark_std::{log2, test_rng};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::{
    expression::generic_expr::Expression,
    layer::{
        batched::{combine_zero_mle_ref, BatchedLayer},
        from_mle, LayerId,
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
struct SimplestBatchedCircuit<F: FieldExt> {
    batched_first_second_mle: Vec<DenseMle<F, Tuple2<F>>>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- Grab combined
        let mut combined_batched_first_second_mle =
            DenseMle::<F, Tuple2<F>>::combine_mle_batch(self.batched_first_second_mle.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut combined_batched_first_second_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        let num_dataparallel_circuit_copies = self.batched_first_second_mle.len();
        let num_dataparallel_bits = log2(num_dataparallel_circuit_copies) as usize;

        // --- Create Layers to be added to ---
        let mut layers: Layers<F, Self::Sponge> = Layers::new();

        // --- Create a SimpleLayer from the first `mle` within the circuit ---
        let diff_builders = self
            .batched_first_second_mle
            .iter_mut()
            .map(|mle| {
                // --- First add batching bits to the MLE (this is a hacky fix and will be removed) ---
                mle.set_prefix_bits(Some(
                    combined_batched_first_second_mle
                        .get_prefix_bits()
                        .iter()
                        .flatten()
                        .cloned()
                        .chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits))
                        .collect_vec(),
                ));

                from_mle(
                    mle,
                    // --- The expression is a simple diff between the first and second halves ---
                    |mle| {
                        let first_half = Expression::mle(mle.first());
                        let second_half = Expression::mle(mle.second());
                        first_half - second_half
                    },
                    // --- The witness generation simply zips the two halves and subtracts them ---
                    |mle, layer_id, prefix_bits| {
                        let num_vars = max(mle.first().num_vars(), mle.second().num_vars());
                        ZeroMleRef::new(num_vars, prefix_bits, layer_id)
                    },
                )
            })
            .collect_vec();

        // --- Convert the vector of builders into a batched builder which can be added to `layers` ---
        let batched_builder = BatchedLayer::new(diff_builders);
        let batched_result = layers.add_gkr(batched_builder);
        let batched_zero = combine_zero_mle_ref(batched_result);

        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_layer: PublicInputLayer<F, Self::Sponge> = input_layer_builder.to_input_layer();

        Witness {
            layers,
            output_layers: vec![batched_zero.get_enum()],
            input_layers: vec![input_layer.to_enum()],
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

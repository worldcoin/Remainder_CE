use ark_std::log2;
use itertools::{multizip, repeat_n, Itertools};
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt};

use crate::{
    layer::{
        batched::{combine_zero_mle_ref, BatchedLayer},
        LayerBuilder, LayerId,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        input_layer::{
            combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
    zkdt::structs::{BinDecomp16Bit, DecisionNode, InputAttribute},
};

use super::circuit_builders::{
    BinaryRecompBuilder, BinaryRecompCheckerBuilder, NodePathDiffBuilder,
};

/// Batched version of the binary recomposition circuit (see below)!
pub struct BinaryRecompCircuitMultiTree<F: FieldExt> {
    batched_decision_node_path_tree_mle: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
    batched_permuted_inputs_tree_mle: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
    batched_diff_signed_bin_decomp_tree_mle: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuitMultiTree<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by dataparallel index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits], [dataparallel_bits], [mle_idx], [iterated_bits] ---
        let combined_batched_decision_node_path_mle_vec = self
            .batched_decision_node_path_tree_mle
            .iter_mut()
            .map(|batched_decision_node_path_mle| {
                DenseMle::<F, DecisionNode<F>>::combine_mle_batch(
                    batched_decision_node_path_mle.clone(),
                )
            })
            .collect_vec();
        let mut combined_batched_decision_node_path_mle =
            DenseMle::<F, F>::combine_mle_batch(combined_batched_decision_node_path_mle_vec);

        let combined_batched_permuted_inputs_mle_vec = self
            .batched_permuted_inputs_tree_mle
            .iter_mut()
            .map(|batched_permuted_inputs_mle| {
                DenseMle::<F, InputAttribute<F>>::combine_mle_batch(
                    batched_permuted_inputs_mle.clone(),
                )
            })
            .collect_vec();
        let mut combined_batched_permuted_inputs_mle =
            DenseMle::<F, F>::combine_mle_batch(combined_batched_permuted_inputs_mle_vec);

        let combined_batched_diff_signed_bin_decomp_mle_vec = self
            .batched_diff_signed_bin_decomp_tree_mle
            .iter_mut()
            .map(|batched_diff_signed_bin_decomp_mle| {
                DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(
                    batched_diff_signed_bin_decomp_mle.clone(),
                )
            })
            .collect_vec();
        let mut combined_batched_diff_signed_bin_decomp_mle =
            DenseMle::<F, F>::combine_mle_batch(combined_batched_diff_signed_bin_decomp_mle_vec);

        // --- Inputs to the circuit are just these three MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut combined_batched_decision_node_path_mle),
            Box::new(&mut combined_batched_permuted_inputs_mle),
            Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
        ];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- NOTE: There is no input layer creation, since this gets handled in the large circuit ---
        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinaryRecompCircuitMultiTree<F> as GKRCircuit<F>>::Sponge> =
            Layers::new();

        let num_tree_bits = log2(self.batched_decision_node_path_tree_mle.len()) as usize;
        let num_subcircuit_copies = self.batched_decision_node_path_tree_mle[0].len() as usize;
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;
        let batched_diff_signed_bin_decomp_mle_prefix_bits =
            self.batched_diff_signed_bin_decomp_tree_mle[0][0].get_prefix_bits();

        let builder_vec = multizip((
            self.batched_decision_node_path_tree_mle.clone().into_iter(),
            self.batched_permuted_inputs_tree_mle.clone().into_iter(),
            self.batched_diff_signed_bin_decomp_tree_mle
                .clone()
                .into_iter(),
        ))
        .map(
            |(
                mut batched_decision_node_path_mle,
                mut batched_permuted_inputs_mle,
                mut batched_diff_signed_bin_decomp_mle,
            )| {
                // --- First we create the positive binary recomp builder ---
                let pos_bin_recomp_builders = batched_diff_signed_bin_decomp_mle
                    .iter_mut()
                    .map(|diff_signed_bit_decomp_mle| {
                        // --- Prefix bits should be [input_prefix_bits], [dataparallel_bits] ---
                        // TODO!(ryancao): Note that strictly speaking we shouldn't be adding dataparallel bits but need to for
                        // now for a specific batching scenario
                        diff_signed_bit_decomp_mle.set_prefix_bits(Some(
                            diff_signed_bit_decomp_mle
                                .get_prefix_bits()
                                .iter()
                                .flatten()
                                .cloned()
                                .chain(repeat_n(
                                    MleIndex::Iterated,
                                    num_dataparallel_bits + num_tree_bits,
                                ))
                                .collect_vec(),
                        ));
                        BinaryRecompBuilder::new(diff_signed_bit_decomp_mle.clone())
                    })
                    .collect();

                let batched_bin_recomp_builder = BatchedLayer::new(pos_bin_recomp_builders);

                // --- Next, we create the diff builder ---
                let batched_diff_builder = BatchedLayer::new(
                    batched_decision_node_path_mle
                        .iter_mut()
                        .zip(batched_permuted_inputs_mle.iter_mut())
                        .map(|(decision_node_path_mle, permuted_inputs_mle)| {
                            // --- Add prefix bits and batching bits to both (same comment as above) ---
                            decision_node_path_mle.set_prefix_bits(Some(
                                decision_node_path_mle
                                    .get_prefix_bits()
                                    .iter()
                                    .flatten()
                                    .cloned()
                                    .chain(repeat_n(
                                        MleIndex::Iterated,
                                        num_dataparallel_bits + num_tree_bits,
                                    ))
                                    .collect_vec(),
                            ));
                            permuted_inputs_mle.set_prefix_bits(Some(
                                permuted_inputs_mle
                                    .get_prefix_bits()
                                    .iter()
                                    .flatten()
                                    .cloned()
                                    .chain(repeat_n(
                                        MleIndex::Iterated,
                                        num_dataparallel_bits + num_tree_bits,
                                    ))
                                    .collect_vec(),
                            ));

                            NodePathDiffBuilder::new(
                                decision_node_path_mle.clone(),
                                permuted_inputs_mle.clone(),
                            )
                        })
                        .collect_vec(),
                );

                // --- Concatenate the previous two builders and add them to the circuit ---
                // let builder = batched_bin_recomp_builder.concat(batched_diff_builder);
                // builder
                (batched_bin_recomp_builder, batched_diff_builder)
            },
        )
        .collect_vec();

        let (batched_bin_recomp_builder_vec, batched_diff_builder_vec): (Vec<_>, Vec<_>) =
            builder_vec.into_iter().unzip();
        let tree_batched_concat_builder = BatchedLayer::new(batched_bin_recomp_builder_vec)
            .concat(BatchedLayer::new(batched_diff_builder_vec));
        let (multi_tree_batched_pos_bin_recomp_mle, multi_tree_batched_raw_diff_mle): (
            Vec<_>,
            Vec<_>,
        ) = layers.add_gkr(tree_batched_concat_builder); //.into_iter().unzip();

        let batched_recomp_checker_builder_vec = multizip((
            self.batched_diff_signed_bin_decomp_tree_mle
                .clone()
                .into_iter(),
            multi_tree_batched_pos_bin_recomp_mle.into_iter(),
            multi_tree_batched_raw_diff_mle.into_iter(),
        ))
        .map(
            |(
                mut batched_diff_signed_bin_decomp_mle,
                batched_pos_bin_recomp_mle,
                batched_raw_diff_mle,
            )| {
                // --- Finally, the recomp checker ---
                let batched_recomp_checker_builder = BatchedLayer::new(
                    batched_diff_signed_bin_decomp_mle
                        .iter_mut()
                        .zip(
                            batched_pos_bin_recomp_mle
                                .into_iter()
                                .zip(batched_raw_diff_mle.into_iter()),
                        )
                        .map(
                            |(diff_signed_bit_decomp_mle, (pos_bin_recomp_mle, raw_diff_mle))| {
                                // --- Add prefix bits to the thing which was indexed earlier ---
                                diff_signed_bit_decomp_mle.set_prefix_bits(Some(
                                    batched_diff_signed_bin_decomp_mle_prefix_bits
                                        .iter()
                                        .flatten()
                                        .cloned()
                                        .chain(repeat_n(
                                            MleIndex::Iterated,
                                            num_dataparallel_bits + num_tree_bits,
                                        ))
                                        .collect_vec(),
                                ));

                                BinaryRecompCheckerBuilder::new(
                                    raw_diff_mle,
                                    diff_signed_bit_decomp_mle.clone(),
                                    pos_bin_recomp_mle,
                                )
                            },
                        )
                        .collect_vec(),
                );

                batched_recomp_checker_builder
            },
        )
        .collect_vec();

        let tree_batched_recomp_checker = BatchedLayer::new(batched_recomp_checker_builder_vec);
        let vec_res_mle = layers.add_gkr(tree_batched_recomp_checker);

        let combined_output_zero_ref = combine_zero_mle_ref(
            vec_res_mle
                .into_iter()
                .map(|inner_zero_vec| combine_zero_mle_ref(inner_zero_vec))
                .collect_vec(),
        );

        let live_committed_input_layer: LigeroInputLayer<F, Self::Sponge> =
            input_layer_builder.to_input_layer_with_rho_inv(4_u8, 1_f64);

        Witness {
            layers,
            output_layers: vec![combined_output_zero_ref.get_enum()],
            input_layers: vec![live_committed_input_layer.to_enum()],
        }
    }
}
impl<F: FieldExt> BinaryRecompCircuitMultiTree<F> {
    /// Creates a new instance of BinaryRecompCircuitBatched
    pub fn new(
        batched_decision_node_path_tree_mle: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
        batched_permuted_inputs_tree_mle: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
        batched_diff_signed_bin_decomp_tree_mle: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
    ) -> Self {
        Self {
            batched_decision_node_path_tree_mle,
            batched_permuted_inputs_tree_mle,
            batched_diff_signed_bin_decomp_tree_mle,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(
        &mut self,
    ) -> Witness<F, <BinaryRecompCircuitMultiTree<F> as GKRCircuit<F>>::Sponge> {
        // --- NOTE: There is no input layer creation, since this gets handled in the large circuit ---
        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinaryRecompCircuitMultiTree<F> as GKRCircuit<F>>::Sponge> =
            Layers::new();

        let num_tree_bits = log2(self.batched_decision_node_path_tree_mle.len()) as usize;
        let num_subcircuit_copies = self.batched_decision_node_path_tree_mle[0].len() as usize;
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        self.batched_decision_node_path_tree_mle
            .iter_mut()
            .for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.set_prefix_bits(Some(
                        mle.get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(
                                MleIndex::Iterated,
                                num_dataparallel_bits + num_tree_bits,
                            ))
                            .collect_vec(),
                    ));
                })
            });

        self.batched_permuted_inputs_tree_mle
            .iter_mut()
            .for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.set_prefix_bits(Some(
                        mle.get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(
                                MleIndex::Iterated,
                                num_dataparallel_bits + num_tree_bits,
                            ))
                            .collect_vec(),
                    ));
                })
            });

        self.batched_diff_signed_bin_decomp_tree_mle
            .iter_mut()
            .for_each(|mle_vec| {
                mle_vec.iter_mut().for_each(|mle| {
                    mle.set_prefix_bits(Some(
                        mle.get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(
                                MleIndex::Iterated,
                                num_dataparallel_bits + num_tree_bits,
                            ))
                            .collect_vec(),
                    ));
                })
            });

        let builder_vec = multizip((
            self.batched_decision_node_path_tree_mle.clone().into_iter(),
            self.batched_permuted_inputs_tree_mle.clone().into_iter(),
            self.batched_diff_signed_bin_decomp_tree_mle
                .clone()
                .into_iter(),
        ))
        .map(
            |(
                mut batched_decision_node_path_mle,
                mut batched_permuted_inputs_mle,
                mut batched_diff_signed_bin_decomp_mle,
            )| {
                // --- First we create the positive binary recomp builder ---
                let pos_bin_recomp_builders = batched_diff_signed_bin_decomp_mle
                    .iter_mut()
                    .map(|diff_signed_bit_decomp_mle| {
                        // --- Prefix bits should be [input_prefix_bits], [dataparallel_bits] ---
                        // TODO!(ryancao): Note that strictly speaking we shouldn't be adding dataparallel bits but need to for
                        // now for a specific batching scenario
                        BinaryRecompBuilder::new(diff_signed_bit_decomp_mle.clone())
                    })
                    .collect();

                let batched_bin_recomp_builder = BatchedLayer::new(pos_bin_recomp_builders);

                // --- Next, we create the diff builder ---
                let batched_diff_builder = BatchedLayer::new(
                    batched_decision_node_path_mle
                        .iter_mut()
                        .zip(batched_permuted_inputs_mle.iter_mut())
                        .map(|(decision_node_path_mle, permuted_inputs_mle)| {
                            NodePathDiffBuilder::new(
                                decision_node_path_mle.clone(),
                                permuted_inputs_mle.clone(),
                            )
                        })
                        .collect_vec(),
                );

                (batched_bin_recomp_builder, batched_diff_builder)

                // --- Concatenate the previous two builders and add them to the circuit ---
                // let builder = batched_bin_recomp_builder.concat(batched_diff_builder);
                // builder
            },
        )
        .collect_vec();

        let (batched_bin_recomp_builder_vec, batched_diff_builder_vec): (Vec<_>, Vec<_>) =
            builder_vec.into_iter().unzip();

        let tree_batched_concat_builder = BatchedLayer::new(batched_bin_recomp_builder_vec)
            .concat(BatchedLayer::new(batched_diff_builder_vec));
        let (multi_tree_batched_pos_bin_recomp_mle, multi_tree_batched_raw_diff_mle): (
            Vec<_>,
            Vec<_>,
        ) = layers.add_gkr(tree_batched_concat_builder); //.into_iter().unzip();
                                                         // dbg!(&multi_tree_batched_pos_bin_recomp_mle[0][0]);
                                                         // dbg!(&multi_tree_batched_raw_diff_mle[0][0]);

        let batched_recomp_checker_builder_vec = multizip((
            self.batched_diff_signed_bin_decomp_tree_mle
                .clone()
                .into_iter(),
            multi_tree_batched_pos_bin_recomp_mle.into_iter(),
            multi_tree_batched_raw_diff_mle.into_iter(),
        ))
        .map(
            |(
                mut batched_diff_signed_bin_decomp_mle,
                batched_pos_bin_recomp_mle,
                batched_raw_diff_mle,
            )| {
                // --- Finally, the recomp checker ---
                let batched_recomp_checker_builder = BatchedLayer::new(
                    batched_diff_signed_bin_decomp_mle
                        .iter_mut()
                        .zip(
                            batched_pos_bin_recomp_mle
                                .into_iter()
                                .zip(batched_raw_diff_mle.into_iter()),
                        )
                        .map(
                            |(diff_signed_bit_decomp_mle, (pos_bin_recomp_mle, raw_diff_mle))| {
                                BinaryRecompCheckerBuilder::new(
                                    raw_diff_mle,
                                    diff_signed_bit_decomp_mle.clone(),
                                    pos_bin_recomp_mle,
                                )
                            },
                        )
                        .collect_vec(),
                );

                batched_recomp_checker_builder
            },
        )
        .collect_vec();

        let tree_batched_recomp_checker = BatchedLayer::new(batched_recomp_checker_builder_vec);
        let vec_res_mle = layers.add_gkr(tree_batched_recomp_checker);

        let combined_output_zero_ref = combine_zero_mle_ref(
            vec_res_mle
                .into_iter()
                .map(|inner_zero_vec| combine_zero_mle_ref(inner_zero_vec))
                .collect_vec(),
        );

        Witness {
            layers,
            output_layers: vec![combined_output_zero_ref.get_enum()],
            input_layers: vec![],
        }
    }
}

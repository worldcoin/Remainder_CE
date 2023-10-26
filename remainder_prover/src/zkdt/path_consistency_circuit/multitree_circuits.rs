use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, MleIndex, mle_enum::MleEnum}, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit}, builders::{ZeroBuilder}}, prover::{GKRCircuit, Witness, Layers, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, enum_input_layer::InputLayerEnum, InputLayer}}, layer::{LayerId, batched::{BatchedLayer, unbatch_mles}}};

use super::{circuit_builders::{OneMinusSignBit, SignBit, PrevNodeLeftBuilderDecision, PrevNodeRightBuilderDecision, CurrNodeBuilderDecision, CurrNodeBuilderLeaf, TwoTimesBuilder, SubtractBuilder}, circuits::{decision_add_wiring_from_size, leaf_add_wiring_from_size}};


/// Same as above, but batched version!
pub struct PathCheckCircuitBatchedNoMulMultiTree<F: FieldExt> {
    batched_decision_node_paths_mle_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
    batched_leaf_node_paths_mle_vec: Vec<Vec<DenseMle<F, LeafNode<F>>>>,
    batched_bin_decomp_diff_mle_vec: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuitBatchedNoMulMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }
}
impl<F: FieldExt> PathCheckCircuitBatchedNoMulMultiTree<F> {
    /// Constructor
    pub fn new(
        batched_decision_node_paths_mle_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
        batched_leaf_node_paths_mle_vec: Vec<Vec<DenseMle<F, LeafNode<F>>>>,
        batched_bin_decomp_diff_mle_vec: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
    ) -> Self {
        Self {
            batched_decision_node_paths_mle_vec,
            batched_leaf_node_paths_mle_vec,
            batched_bin_decomp_diff_mle_vec
        }
    }

    /// To be used in the large combined circuit. Note that we cannot directly
    /// call `add_gkr()` on gate MLE layers, so we must manually add each layer
    /// from this subcircuit into the `combined_layers` parameter.
    /// 
    /// ## Arguments
    /// * `combined_layers` - The layers from the combined circuit we are adding to
    /// * `combined_output_layers` - The output layers from the combined circuit we are adding to
    /// 
    /// ## Returns
    /// * `new_combined_output_layers` - The original `combined_output_layers`, but with
    ///     output layers generated from this subcircuit appended.
    pub fn add_subcircuit_layers_to_combined_layers(&mut self, 
        combined_layers: &mut Layers<F, PoseidonTranscript<F>>,
        combined_output_layers: Vec<MleEnum<F>>,
    ) -> Vec<MleEnum<F>> {
        let num_tree_bits = log2(self.batched_decision_node_paths_mle_vec.len()) as usize;
        let num_dataparallel_circuits = self.batched_decision_node_paths_mle_vec[0].len();
        let num_dataparallel_bits = log2(num_dataparallel_circuits) as usize;


        self.batched_decision_node_paths_mle_vec.iter_mut().for_each(
            |mle_vec| { mle_vec.iter_mut().for_each(
                    |mle| { mle.set_prefix_bits(Some(
                            mle.get_prefix_bits().iter().flatten().cloned().chain(
                                repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)
                            ).collect_vec()
                        ));
                    });
            });

        self.batched_leaf_node_paths_mle_vec.iter_mut().for_each(
            |mle_vec| { mle_vec.iter_mut().for_each(
                    |mle| { mle.set_prefix_bits(Some(
                            mle.get_prefix_bits().iter().flatten().cloned().chain(
                                repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)
                            ).collect_vec()
                        ));
                    });
            });
        
        self.batched_bin_decomp_diff_mle_vec.iter_mut().for_each(
            |mle_vec| { mle_vec.iter_mut().for_each(
                    |mle| { mle.set_prefix_bits(Some(
                            mle.get_prefix_bits().iter().flatten().cloned().chain(
                                repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)
                            ).collect_vec()
                        ));
                    });
            });

        let pos_builders_vec = self.batched_bin_decomp_diff_mle_vec.iter().map(
            |batched_bin_decomp_diff_mle| {
                let pos_builders = batched_bin_decomp_diff_mle.iter().map(
                    |bin_decomp_mle| {
                        OneMinusSignBit::new(bin_decomp_mle.clone())
                    }
                ).collect_vec();
                BatchedLayer::new(pos_builders)
            }
        ).collect_vec();

        let pos_sign_bits_vec = combined_layers.add_gkr(BatchedLayer::new(pos_builders_vec)); // ID is 0


        let two_times_batched_builder_vec = self.batched_decision_node_paths_mle_vec.iter().map(
            |batched_decision_node_paths_mle| {
                let two_times_builders = batched_decision_node_paths_mle.iter().map(
                    |dec_mle| {
                        TwoTimesBuilder::new(dec_mle.clone())
                    }
                ).collect_vec();
        
                let two_times_batched_builder = BatchedLayer::new(two_times_builders);
                two_times_batched_builder
            }
        ).collect_vec();

        let two_times_things_vec = combined_layers.add_gkr(BatchedLayer::new(two_times_batched_builder_vec));

        
        let two_times_plus_sign_builders_vec = two_times_things_vec.iter().zip(pos_sign_bits_vec.iter()).map(
            |(two_times_things, pos_sign_bits)| {
                let two_times_plus_sign_builders = two_times_things.iter().zip(pos_sign_bits.iter()).map(
                    |(two_times, sign_bit)| {
                        SubtractBuilder::new(two_times.clone(), sign_bit.clone())
                    }
                ).collect_vec();
                let two_times_plus_sign_builder = BatchedLayer::new(two_times_plus_sign_builders);
                two_times_plus_sign_builder
            }
        ).collect_vec();
        
        let two_times_plus_sign_vals_vec = combined_layers.add_gkr(BatchedLayer::new(two_times_plus_sign_builders_vec));


        let curr_node_decision_builders_vec = self.batched_decision_node_paths_mle_vec.iter().map(
            |batched_decision_node_paths_mle| {
                let curr_node_decision_builders = batched_decision_node_paths_mle.iter().map(
                    |dec_mle| {
                        CurrNodeBuilderDecision::new(dec_mle.clone())
                    }
                ).collect_vec();
        
                let curr_decision_batched_builder = BatchedLayer::new(curr_node_decision_builders);
                curr_decision_batched_builder
            }
        ).collect_vec();

        let curr_node_leaf_builders_vec = self.batched_leaf_node_paths_mle_vec.iter().map(
            |batched_leaf_node_paths_mle| {
                let curr_node_leaf_builders = batched_leaf_node_paths_mle.iter().map(
                    |leaf_mle| {
                        CurrNodeBuilderLeaf::new(leaf_mle.clone())
                    }
                ).collect_vec();
        
                let curr_leaf_batched_builder = BatchedLayer::new(curr_node_leaf_builders);
                curr_leaf_batched_builder
            }
        ).collect_vec();

        let curr_decision_vec = combined_layers.add_gkr(BatchedLayer::new(curr_node_decision_builders_vec)); // ID is 2
        let curr_leaf_vec = combined_layers.add_gkr(BatchedLayer::new(curr_node_leaf_builders_vec)); // ID is 3

        let flattened_curr_dec_vec = curr_decision_vec.into_iter().map(|curr_decision| unbatch_mles(curr_decision)).collect_vec();
        let flattened_curr_leaf_vec = curr_leaf_vec.into_iter().map(|curr_leaf| unbatch_mles(curr_leaf)).collect_vec();

        let flattened_curr_dec = unbatch_mles(flattened_curr_dec_vec);
        let flattened_curr_leaf = unbatch_mles(flattened_curr_leaf_vec);

        let two_times_plus_sign_vals = two_times_plus_sign_vals_vec.into_iter().map(|two_times_plus_sign_vals| unbatch_mles(two_times_plus_sign_vals)).collect_vec();
        let flattened_two_times_plus_sign = unbatch_mles(two_times_plus_sign_vals);

        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));

        let res_dec = combined_layers.add_add_gate_batched(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_two_times_plus_sign.mle_ref(), num_dataparallel_bits); // ID is 6

        let res_leaf = combined_layers.add_add_gate_batched(nonzero_gates_add_leaf.clone(), flattened_two_times_plus_sign.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 8

        let res_dec_zero = ZeroBuilder::new(res_dec);
        let res_leaf_zero = ZeroBuilder::new(res_leaf);

        let res_dec_zero_mle = combined_layers.add_gkr(res_dec_zero); // ID is 14
        let res_leaf_zero_mle = combined_layers.add_gkr(res_leaf_zero); // ID is 15

        let new_output_layers = vec![res_dec_zero_mle.get_enum(), res_leaf_zero_mle.get_enum()];

        combined_output_layers.into_iter().chain(new_output_layers.into_iter()).collect_vec()

    }
}
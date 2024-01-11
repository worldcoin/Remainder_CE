use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, MleIndex, mle_enum::MleEnum}, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit}, builders::{ZeroBuilder}}, prover::{GKRCircuit, Witness, Layers, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, enum_input_layer::InputLayerEnum, InputLayer}}, layer::{LayerId, batched::{BatchedLayer, unbatch_mles}}, gate::gate::BinaryOperation};

use super::circuit_builders::{OneMinusSignBit, SignBit, PrevNodeLeftBuilderDecision, PrevNodeRightBuilderDecision, CurrNodeBuilderDecision, CurrNodeBuilderLeaf, TwoTimesBuilder, SubtractBuilder};

/// Given an AddGate or AddGateBatched mle struct with `lhs` and `rhs` (on either side
/// of the summation) which are both mles over decision nodes, this creates the wiring 
/// between the two sides for the current circuit.
/// 
/// # Arguments
/// `size`: the maximum of the number of elements in the bookkeeping table in the 
/// AddGate/AddGateBatched between `lhs` and `rhs`
/// 
/// # Returns
/// a vector of gates which are nonzero representing the wiring specifically for 
/// adding the current node id to the expected next node id for decision nodes.
pub fn decision_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    
    (0 .. (size-1)).map(
        |idx| (idx, idx + 1, idx)
    ).collect_vec()
}

/// Given an AddGate or AddGateBatched mle struct with `lhs` and `rhs` (on either side
/// of the summation) where one side is decision nodes and the other a leaf node, this 
/// creates the wiring between the two sides for the current circuit.
/// 
/// # Arguments
/// `size`: the maximum of the number of elements in the bookkeeping table in the 
/// AddGate/AddGateBatched between `lhs` and `rhs`
/// 
/// # Returns
/// a vector of nonzero gates representing the wiring specifically for adding the current 
/// node id to the expected next node id for leaf nodes against the last decision node.
pub fn leaf_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size-1, 0)]
}


/// Given an MulGate or MulGateBatched mle struct with `lhs` and `rhs` (on either side
/// of the product) which one is over decision nodes and the other are sign bits, this 
/// creates the wiring between the two sides for the current circuit.
/// 
/// # Arguments
/// `size`: the maximum of the number of elements in the bookkeeping table in the 
/// MulGate/MulGateBatched between `lhs` and `rhs`
/// 
/// # Returns
/// a vector of gates which are nonzero representing the wiring specifically for 
/// multiplying the current decision node id to the respective sign bit.
pub fn decision_mul_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    dbg!(&size);
    
    (0 .. (size-1)).map(
        |idx| (idx, idx, idx)
    ).collect_vec()
}

/// Given an MulGate or MulGateBatched mle struct with `lhs` and `rhs` (on either side
/// of the product) which one is over leaf nodes and the other are sign bits, this 
/// creates the wiring between the two sides for the current circuit.
/// 
/// # Arguments
/// `size`: the maximum of the number of elements in the bookkeeping table in the 
/// MulGate/MulGateBatched between `lhs` and `rhs`
/// 
/// # Returns
/// a vector of gates which are nonzero representing the wiring specifically for 
/// multiplying the current leaf node id to the last sign bit of the decision nodes.
pub fn leaf_mul_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size-1, 0)]
}

/// Same as above, but batched version!
pub struct PathCheckCircuitBatched<F: FieldExt> {
    batched_decision_node_paths_mle: Vec<DenseMle<F, DecisionNode<F>>>,
    batched_leaf_node_paths_mle: Vec<DenseMle<F, LeafNode<F>>>,
    batched_bin_decomp_diff_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuitBatched<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let num_dataparallel_circuits = self.batched_decision_node_paths_mle.len();
        let num_dataparallel_bits = log2(num_dataparallel_circuits) as usize;
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        let mut combined_decision = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(self.batched_decision_node_paths_mle.clone());
        let mut combined_leaf = DenseMle::<F, LeafNode<F>>::combine_mle_batch(self.batched_leaf_node_paths_mle.clone());
        let mut combined_bit = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.batched_bin_decomp_diff_mle.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_decision), Box::new(&mut combined_leaf), Box::new(&mut combined_bit)];
        let input_layer_builder = InputLayerBuilder::<F>::new(input_mles, None, LayerId::Input(0));
        let input_layer: InputLayerEnum<F, Self::Transcript> = input_layer_builder.to_input_layer::<PublicInputLayer<F, _>>().to_enum();

        self.batched_bin_decomp_diff_mle.iter_mut().for_each(
            |bin_decomp_mle| {
                bin_decomp_mle.set_prefix_bits(Some(
                    combined_bit.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        self.batched_decision_node_paths_mle.iter_mut().for_each(
            |dec_mle| {
                dec_mle.set_prefix_bits(Some(
                    combined_decision.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        self.batched_leaf_node_paths_mle.iter_mut().for_each(
            |leaf_mle| {
                leaf_mle.set_prefix_bits(Some(
                    combined_leaf.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        let pos_builders = self.batched_bin_decomp_diff_mle.iter().map(
            |bin_decomp_mle| {
                OneMinusSignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let pos_batched_builder = BatchedLayer::new(pos_builders);


        let pos_sign_bits = layers.add_gkr(pos_batched_builder); 

        let two_times_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                TwoTimesBuilder::new(dec_mle.clone())
            }
        ).collect_vec();

        let two_times_batched_builder = BatchedLayer::new(two_times_builders);
        let two_times_things = layers.add_gkr(two_times_batched_builder);

        let two_times_plus_sign_builders = two_times_things.iter().zip(pos_sign_bits.iter()).map(
            |(two_times, sign_bit)| {
                SubtractBuilder::new(two_times.clone(), sign_bit.clone())
            }
        ).collect_vec();
        let two_times_plus_sign_builder = BatchedLayer::new(two_times_plus_sign_builders);
        let two_times_plus_sign_vals = layers.add_gkr(two_times_plus_sign_builder);


        let curr_node_decision_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                CurrNodeBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let curr_decision_batched_builder = BatchedLayer::new(curr_node_decision_builders);

        let curr_node_leaf_builders = self.batched_leaf_node_paths_mle.iter().map(
            |leaf_mle| {
                CurrNodeBuilderLeaf::new(leaf_mle.clone())
            }
        ).collect_vec();

        let curr_leaf_batched_builder = BatchedLayer::new(curr_node_leaf_builders);

        let curr_decision = layers.add_gkr(curr_decision_batched_builder); 
        let curr_leaf = layers.add_gkr(curr_leaf_batched_builder); 

        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
        let flattened_two_times_plus_sign = unbatch_mles(two_times_plus_sign_vals);

        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));

        let res_dec = layers.add_gate(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_two_times_plus_sign.mle_ref(), num_dataparallel_bits, BinaryOperation::Add); 

        let res_leaf = layers.add_gate(nonzero_gates_add_leaf.clone(), flattened_two_times_plus_sign.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits, BinaryOperation::Add); 

        let res_dec_zero = ZeroBuilder::new(res_dec);
        let res_leaf_zero = ZeroBuilder::new(res_leaf);

        let res_dec_zero_mle = layers.add_gkr(res_dec_zero); 
        let res_leaf_zero_mle = layers.add_gkr(res_leaf_zero); 

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![res_dec_zero_mle.get_enum(), res_leaf_zero_mle.get_enum()],
            input_layers: vec![input_layer]
        };

        witness
    }
}
impl<F: FieldExt> PathCheckCircuitBatched<F> {
    /// Constructor
    pub fn new(
        batched_decision_node_paths_mle: Vec<DenseMle<F, DecisionNode<F>>>,
        batched_leaf_node_paths_mle: Vec<DenseMle<F, LeafNode<F>>>,
        batched_bin_decomp_diff_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            batched_decision_node_paths_mle,
            batched_leaf_node_paths_mle,
            batched_bin_decomp_diff_mle
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
        let num_dataparallel_circuits = self.batched_decision_node_paths_mle.len();
        let num_dataparallel_bits = log2(num_dataparallel_circuits) as usize;

        self.batched_bin_decomp_diff_mle.iter_mut().for_each(
            |bin_decomp_mle| {
                bin_decomp_mle.set_prefix_bits(Some(
                    bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        self.batched_decision_node_paths_mle.iter_mut().for_each(
            |dec_mle| {
                dec_mle.set_prefix_bits(Some(
                    dec_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        self.batched_leaf_node_paths_mle.iter_mut().for_each(
            |leaf_mle| {
                leaf_mle.set_prefix_bits(Some(
                    leaf_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                ));
            }
        );

        let pos_builders = self.batched_bin_decomp_diff_mle.iter().map(
            |bin_decomp_mle| {
                OneMinusSignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let pos_batched_builder = BatchedLayer::new(pos_builders);


        let pos_sign_bits = combined_layers.add_gkr(pos_batched_builder); 

        let two_times_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                TwoTimesBuilder::new(dec_mle.clone())
            }
        ).collect_vec();

        let two_times_batched_builder = BatchedLayer::new(two_times_builders);
        let two_times_things = combined_layers.add_gkr(two_times_batched_builder);

        let two_times_plus_sign_builders = two_times_things.iter().zip(pos_sign_bits.iter()).map(
            |(two_times, sign_bit)| {
                SubtractBuilder::new(two_times.clone(), sign_bit.clone())
            }
        ).collect_vec();
        let two_times_plus_sign_builder = BatchedLayer::new(two_times_plus_sign_builders);
        let two_times_plus_sign_vals = combined_layers.add_gkr(two_times_plus_sign_builder);


        let curr_node_decision_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                CurrNodeBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let curr_decision_batched_builder = BatchedLayer::new(curr_node_decision_builders);

        let curr_node_leaf_builders = self.batched_leaf_node_paths_mle.iter().map(
            |leaf_mle| {
                CurrNodeBuilderLeaf::new(leaf_mle.clone())
            }
        ).collect_vec();

        let curr_leaf_batched_builder = BatchedLayer::new(curr_node_leaf_builders);

        let curr_decision = combined_layers.add_gkr(curr_decision_batched_builder); 
        let curr_leaf = combined_layers.add_gkr(curr_leaf_batched_builder); 

        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
        let flattened_two_times_plus_sign = unbatch_mles(two_times_plus_sign_vals);

        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));

        let res_dec = combined_layers.add_gate(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_two_times_plus_sign.mle_ref(), num_dataparallel_bits, BinaryOperation::Add); 

        let res_leaf = combined_layers.add_gate(nonzero_gates_add_leaf.clone(), flattened_two_times_plus_sign.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits, BinaryOperation::Add); 

        let res_dec_zero = ZeroBuilder::new(res_dec);
        let res_leaf_zero = ZeroBuilder::new(res_leaf);

        let res_dec_zero_mle = combined_layers.add_gkr(res_dec_zero); 
        let res_leaf_zero_mle = combined_layers.add_gkr(res_leaf_zero); 

        let new_output_layers = vec![res_dec_zero_mle.get_enum(), res_leaf_zero_mle.get_enum()];

        combined_output_layers.into_iter().chain(new_output_layers.into_iter()).collect_vec()

    }
}
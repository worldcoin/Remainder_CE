
use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, MleIndex, mle_enum::MleEnum}, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit}, builders::{ZeroBuilder}}, prover::{GKRCircuit, Witness, Layers, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, enum_input_layer::InputLayerEnum, InputLayer}}, layer::{LayerId, batched::{BatchedLayer, unbatch_mles}}};

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
pub struct PathCheckCircuitBatchedMul<F: FieldExt> {
    batched_decision_node_paths_mle: Vec<DenseMle<F, DecisionNode<F>>>,
    batched_leaf_node_paths_mle: Vec<DenseMle<F, LeafNode<F>>>,
    batched_bin_decomp_diff_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuitBatchedMul<F> {
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

        let neg_builders = self.batched_bin_decomp_diff_mle.iter().map(
            |bin_decomp_mle| {
                SignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let neg_batched_builder = BatchedLayer::new(neg_builders);

        let pos_sign_bits = layers.add_gkr(pos_batched_builder); // ID is 0
        let neg_sign_bits = layers.add_gkr(neg_batched_builder); // ID is 1

        let prev_node_left_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                PrevNodeLeftBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_left_batched_builder = BatchedLayer::new(prev_node_left_builders);

        let prev_node_right_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                PrevNodeRightBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_right_batched_builder = BatchedLayer::new(prev_node_right_builders);

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

        let curr_decision = layers.add_gkr(curr_decision_batched_builder); // ID is 2
        let curr_leaf = layers.add_gkr(curr_leaf_batched_builder); // ID is 3
        let prev_node_right = layers.add_gkr(prev_right_batched_builder); // ID is 4
        let prev_node_left = layers.add_gkr(prev_left_batched_builder); // ID is 5

        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
        let flattened_prev_right = unbatch_mles(prev_node_right);
        let flattened_prev_left = unbatch_mles(prev_node_left);


        // add gate with dec and right
        // add gate with dec and left
        // add gate with leaf and right
        // add gate with leaf and left
        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_prev_left.num_iterated_vars() - num_dataparallel_bits));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_prev_left.num_iterated_vars() - num_dataparallel_bits));

        let res_neg_dec = layers.add_add_gate_batched(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_prev_left.mle_ref(), num_dataparallel_bits); // ID is 6
        let res_pos_dec = layers.add_add_gate_batched(nonzero_gates_add_decision, flattened_curr_dec.mle_ref(), flattened_prev_right.mle_ref(), num_dataparallel_bits); // ID is 7

        let res_neg_leaf = layers.add_add_gate_batched(nonzero_gates_add_leaf.clone(), flattened_prev_left.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 8
        let res_pos_leaf = layers.add_add_gate_batched(nonzero_gates_add_leaf, flattened_prev_right.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 9

        let nonzero_gates_mul_decision = decision_mul_wiring_from_size(1 << pos_sign_bits[0].num_iterated_vars());
        let nonzero_gates_mul_leaf = leaf_mul_wiring_from_size(1 << pos_sign_bits[0].num_iterated_vars());

        let flattened_pos = unbatch_mles(pos_sign_bits);
        let flattened_neg = unbatch_mles(neg_sign_bits);

        let dec_pos_prod = layers.add_mul_gate_batched(nonzero_gates_mul_decision.clone(), flattened_pos.mle_ref(), res_pos_dec.mle_ref(), num_dataparallel_bits); // ID is 10
        let dec_neg_prod = layers.add_mul_gate_batched(nonzero_gates_mul_decision, flattened_neg.mle_ref(), res_neg_dec.mle_ref(), num_dataparallel_bits); // ID is 11
        let leaf_pos_prod = layers.add_mul_gate_batched(nonzero_gates_mul_leaf.clone(), flattened_pos.mle_ref(), res_pos_leaf.mle_ref(), num_dataparallel_bits); // ID is 12
        let leaf_neg_prod = layers.add_mul_gate_batched(nonzero_gates_mul_leaf, flattened_neg.mle_ref(), res_neg_leaf.mle_ref(), num_dataparallel_bits); // ID is 13

        let dec_pos_zero = ZeroBuilder::new(dec_pos_prod);
        let dec_neg_zero = ZeroBuilder::new(dec_neg_prod);
        let leaf_pos_zero = ZeroBuilder::new(leaf_pos_prod);
        let leaf_neg_zero = ZeroBuilder::new(leaf_neg_prod);

        let dec_pos_zero_mle = layers.add_gkr(dec_pos_zero); // ID is 14
        let dec_neg_zero_mle = layers.add_gkr(dec_neg_zero); // ID is 15
        let leaf_pos_zero_mle = layers.add_gkr(leaf_pos_zero); // ID is 16
        let leaf_neg_zero_mle = layers.add_gkr(leaf_neg_zero); // ID is 17

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![dec_pos_zero_mle.get_enum(), dec_neg_zero_mle.get_enum(), leaf_pos_zero_mle.get_enum(), leaf_neg_zero_mle.get_enum()],
            input_layers: vec![input_layer]
        };

        witness
    }
}
impl<F: FieldExt> PathCheckCircuitBatchedMul<F> {
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
        
        // get the batch size, or the number of inputs, by getting the length of the vector
        let num_dataparallel_circuits = self.batched_decision_node_paths_mle.len();
        // number of bits needed to represent which copy number we are currently in
        let num_dataparallel_bits = log2(num_dataparallel_circuits) as usize;

        // for each of the mles, we need to add prefix bits corresopnding to the number of dataparallel circuits.
        // we do this manually by adding the number of iterated bits corresponding to the number of dataparallel bits
        // now the prefix bits are in the order (mle ref bits, batched bits)

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

        // we compute (1 - (sign bit)) using this builder in order to get the values of the bit which are nonzero
        // if and only if the threshold difference against the node value is positive
        let pos_builders = self.batched_bin_decomp_diff_mle.iter().map(
            |bin_decomp_mle| {
                OneMinusSignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let pos_batched_builder = BatchedLayer::new(pos_builders);

        // we compute (sign bit) using this builder in order to get the values of the bit which are nonzero
        // if and only if the threshold difference against the node value is negative
        let neg_builders = self.batched_bin_decomp_diff_mle.iter().map(
            |bin_decomp_mle| {
                SignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let neg_batched_builder = BatchedLayer::new(neg_builders);

        // add these layers to be sumchecked over
        let pos_sign_bits = combined_layers.add_gkr(pos_batched_builder); // ID is 0
        let neg_sign_bits = combined_layers.add_gkr(neg_batched_builder); // ID is 1

        // for the decision path mle, compute (2*node_id + 1)
        let prev_node_left_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                PrevNodeLeftBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_left_batched_builder = BatchedLayer::new(prev_node_left_builders);

        // for the decision path mle, compute (2*node_id + 2)
        let prev_node_right_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                PrevNodeRightBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_right_batched_builder = BatchedLayer::new(prev_node_right_builders);

        // for the decision path mle, return mle representing node_id
        let curr_node_decision_builders = self.batched_decision_node_paths_mle.iter().map(
            |dec_mle| {
                CurrNodeBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let curr_decision_batched_builder = BatchedLayer::new(curr_node_decision_builders);

        // for the leaf path mle, return mle representing node_id
        let curr_node_leaf_builders = self.batched_leaf_node_paths_mle.iter().map(
            |leaf_mle| {
                CurrNodeBuilderLeaf::new(leaf_mle.clone())
            }
        ).collect_vec();

        let curr_leaf_batched_builder = BatchedLayer::new(curr_node_leaf_builders);

        // add these layers to be sumchecked over
        let curr_decision = combined_layers.add_gkr(curr_decision_batched_builder); // ID is 2
        let curr_leaf = combined_layers.add_gkr(curr_leaf_batched_builder); // ID is 3
        let prev_node_right = combined_layers.add_gkr(prev_right_batched_builder); // ID is 4
        let prev_node_left = combined_layers.add_gkr(prev_left_batched_builder); // ID is 5

        // in order to use with the gate mles, we need to flatten the vector of mles such that it is one large mle
        // with num_dataparallel_bits + num_iterated_bits number of bits (combined in little endian format)
        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
        let flattened_prev_right = unbatch_mles(prev_node_right);
        let flattened_prev_left = unbatch_mles(prev_node_left);



        // get the circuit wiring for two decision path mles 
        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_prev_left.num_iterated_vars() - num_dataparallel_bits));
        // get circuit wiring for decision path mle + leaf path mle
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_prev_left.num_iterated_vars() - num_dataparallel_bits));

        // we want to compute (id_{j+1}) - (2id_j + 1) where j is the node id. we do this using gate mles because this is
        // an irregular circuit. this is two steps because we do this separately for when we are comparing decision nodes
        // against decision nodes, and decision nodes against the last leaf node.
        // this specific step computes the cases that should be zero when we go left in the decision tree
        let res_neg_dec = combined_layers.add_add_gate_batched(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_prev_left.mle_ref(), num_dataparallel_bits); // ID is 6
        let res_neg_leaf = combined_layers.add_add_gate_batched(nonzero_gates_add_leaf.clone(), flattened_prev_left.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 8

        // we want to compute (id_{j+1}) - (2id_j + 2) where j is the node id. we do this using gate mles because this is
        // an irregular circuit. this is two steps because we do this separately for when we are comparing decision nodes
        // against decision nodes, and decision nodes against the last leaf node.
        // this specific step computes the cases that should be zero when we go right in the decision tree
        let res_pos_dec = combined_layers.add_add_gate_batched(nonzero_gates_add_decision, flattened_curr_dec.mle_ref(), flattened_prev_right.mle_ref(), num_dataparallel_bits); // ID is 7
        let res_pos_leaf = combined_layers.add_add_gate_batched(nonzero_gates_add_leaf, flattened_prev_right.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 9

        // get the circuit wiring for a decision path mle and sign bit mle
        let nonzero_gates_mul_decision = decision_mul_wiring_from_size(1 << pos_sign_bits[0].num_iterated_vars());
        // get the circuit wiring for a leaf path mle and a sign bit mle
        let nonzero_gates_mul_leaf = leaf_mul_wiring_from_size(1 << pos_sign_bits[0].num_iterated_vars());

        // in order to use in the MulGateBatched, we need a flattened mle as described above
        let flattened_pos = unbatch_mles(pos_sign_bits);
        let flattened_neg = unbatch_mles(neg_sign_bits);

        // we want to multiply the sign bits that should be zero when we turn right * the differences that are nonzero when we turn right, and
        // the sign bits that shoudl be zero when we turn left * the differences that are nonzero when we turn left. we do this in four steps
        // because this is done differently when we have decision nodes and the leaf node, due to the circuit wiring.
        let dec_pos_prod = combined_layers.add_mul_gate_batched(nonzero_gates_mul_decision.clone(), flattened_pos.mle_ref(), res_pos_dec.mle_ref(), num_dataparallel_bits); // ID is 10
        let dec_neg_prod = combined_layers.add_mul_gate_batched(nonzero_gates_mul_decision, flattened_neg.mle_ref(), res_neg_dec.mle_ref(), num_dataparallel_bits); // ID is 11
        let leaf_pos_prod = combined_layers.add_mul_gate_batched(nonzero_gates_mul_leaf.clone(), flattened_pos.mle_ref(), res_pos_leaf.mle_ref(), num_dataparallel_bits); // ID is 12
        let leaf_neg_prod = combined_layers.add_mul_gate_batched(nonzero_gates_mul_leaf, flattened_neg.mle_ref(), res_neg_leaf.mle_ref(), num_dataparallel_bits); // ID is 13

        // we return ZeroMleRefs at the end of this process, and therefore add an extra step where we add the mles to a ZeroBuilder
        let dec_pos_zero = ZeroBuilder::new(dec_pos_prod);
        let dec_neg_zero = ZeroBuilder::new(dec_neg_prod);
        let leaf_pos_zero = ZeroBuilder::new(leaf_pos_prod);
        let leaf_neg_zero = ZeroBuilder::new(leaf_neg_prod);

        let dec_pos_zero_mle = combined_layers.add_gkr(dec_pos_zero); // ID is 14
        let dec_neg_zero_mle = combined_layers.add_gkr(dec_neg_zero); // ID is 15
        let leaf_pos_zero_mle = combined_layers.add_gkr(leaf_pos_zero); // ID is 16
        let leaf_neg_zero_mle = combined_layers.add_gkr(leaf_neg_zero); // ID is 17

        // --- Grab output layers and add to original combined circuit output layers ---
        let new_output_layers = vec![dec_pos_zero_mle.get_enum(), dec_neg_zero_mle.get_enum(), leaf_pos_zero_mle.get_enum(), leaf_neg_zero_mle.get_enum()];
        combined_output_layers.into_iter().chain(new_output_layers.into_iter()).collect_vec()

    }
}

/// Same as above, but batched version!
pub struct PathCheckCircuitBatchedNoMul<F: FieldExt> {
    batched_decision_node_paths_mle: Vec<DenseMle<F, DecisionNode<F>>>,
    batched_leaf_node_paths_mle: Vec<DenseMle<F, LeafNode<F>>>,
    batched_bin_decomp_diff_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuitBatchedNoMul<F> {
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


        let pos_sign_bits = layers.add_gkr(pos_batched_builder); // ID is 0

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

        let curr_decision = layers.add_gkr(curr_decision_batched_builder); // ID is 2
        let curr_leaf = layers.add_gkr(curr_leaf_batched_builder); // ID is 3

        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
        let flattened_two_times_plus_sign = unbatch_mles(two_times_plus_sign_vals);

        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (flattened_two_times_plus_sign.num_iterated_vars() - num_dataparallel_bits));

        let res_dec = layers.add_add_gate_batched(nonzero_gates_add_decision.clone(), flattened_curr_dec.mle_ref(), flattened_two_times_plus_sign.mle_ref(), num_dataparallel_bits); // ID is 6

        let res_leaf = layers.add_add_gate_batched(nonzero_gates_add_leaf.clone(), flattened_two_times_plus_sign.mle_ref(), flattened_curr_leaf.mle_ref(), num_dataparallel_bits); // ID is 8

        let res_dec_zero = ZeroBuilder::new(res_dec);
        let res_leaf_zero = ZeroBuilder::new(res_leaf);

        let res_dec_zero_mle = layers.add_gkr(res_dec_zero); // ID is 14
        let res_leaf_zero_mle = layers.add_gkr(res_leaf_zero); // ID is 15

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![res_dec_zero_mle.get_enum(), res_leaf_zero_mle.get_enum()],
            input_layers: vec![input_layer]
        };

        witness
    }
}
impl<F: FieldExt> PathCheckCircuitBatchedNoMul<F> {
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


        let pos_sign_bits = combined_layers.add_gkr(pos_batched_builder); // ID is 0

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

        let curr_decision = combined_layers.add_gkr(curr_decision_batched_builder); // ID is 2
        let curr_leaf = combined_layers.add_gkr(curr_leaf_batched_builder); // ID is 3

        let flattened_curr_dec = unbatch_mles(curr_decision);
        let flattened_curr_leaf = unbatch_mles(curr_leaf);
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
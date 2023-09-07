use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, MleIndex}, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit, combine_mle_refs}, zkdt_layer::ConcatBuilder}, prover::{GKRCircuit, Witness, Layers, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, enum_input_layer::InputLayerEnum, InputLayer}}, layer::{LayerId, empty_layer::EmptyLayer, batched::{combine_mles, BatchedLayer, unbatch_mles, unflatten_mle, combine_zero_mle_ref}}};

use super::circuit_builders::{OneMinusSignBit, SignBit, PrevNodeLeftBuilderDecision, PrevNodeRightBuilderDecision, CurrNodeBuilderDecision, CurrNodeBuilderLeaf, SignBitProductBuilder, DumbBuilder};

/// Helper!
pub fn create_wiring_from_num_bits(num_bits: usize) -> Vec<(usize, usize, usize)> {
    dbg!(&num_bits);
    let mut gates = (0.. (num_bits-1)).into_iter().map(
        |idx| (idx, 2*(idx + 1), idx)
    ).collect_vec();
    gates.push((num_bits-1, 1, num_bits-1));

    gates
}

/// For checking whether a path is consistent (i.e. all children of
/// parent nodes is either 2i + 1 or 2i + 2, where `i` is the parent's
/// node ID)
pub struct PathCheckCircuit<F: FieldExt> {
    decision_node_paths_mle: DenseMle<F, DecisionNode<F>>, 
    leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
    bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
    num_copy: usize,
}

impl<F: FieldExt> GKRCircuit<F> for PathCheckCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.decision_node_paths_mle), Box::new(&mut self.leaf_node_paths_mle), Box::new(&mut self.bin_decomp_diff_mle)];
        let input_layer_builder = InputLayerBuilder::<F>::new(input_mles, None, LayerId::Input(0));
        let input_layer = input_layer_builder.to_input_layer::<PublicInputLayer<F, _>>().to_enum();

        let pos_sign_bit_builder = OneMinusSignBit::new(self.bin_decomp_diff_mle.clone());
        let pos_sign_bits = layers.add_gkr(pos_sign_bit_builder);
       
        let neg_sign_bit_builder = SignBit::new(self.bin_decomp_diff_mle.clone());
        let neg_sign_bits = layers.add_gkr(neg_sign_bit_builder);
        
        let prev_node_left_builder = PrevNodeLeftBuilderDecision::new(
            self.decision_node_paths_mle.clone());

        let prev_node_right_builder = PrevNodeRightBuilderDecision::new(
            self.decision_node_paths_mle.clone());

        let curr_node_decision_builder = CurrNodeBuilderDecision::new(
            self.decision_node_paths_mle.clone());

        let curr_node_leaf_builder = CurrNodeBuilderLeaf::new(
            self.leaf_node_paths_mle.clone());

        let curr_decision = layers.add_gkr(curr_node_decision_builder);
        let curr_leaf = layers.add::<_, EmptyLayer<F, Self::Transcript>>(curr_node_leaf_builder);

        let curr_node_decision_leaf_builder = ConcatBuilder::new(curr_decision.clone(), curr_leaf.clone());
        let curr_node_decision_leaf_mle_ref = layers.add_gkr(curr_node_decision_leaf_builder).mle_ref();
        let prev_node_right_mle_ref = layers.add_gkr(prev_node_right_builder).mle_ref();
        let prev_node_left_mle_ref = layers.add_gkr(prev_node_left_builder).mle_ref();
        
        let nonzero_gates = create_wiring_from_num_bits(1 << (prev_node_left_mle_ref.num_vars() - self.num_copy));
         
        let res_negative = layers.add_add_gate(nonzero_gates.clone(), curr_node_decision_leaf_mle_ref.clone(), prev_node_left_mle_ref.clone(), self.num_copy);
        let res_positive = layers.add_add_gate(nonzero_gates, curr_node_decision_leaf_mle_ref, prev_node_right_mle_ref.clone(), self.num_copy);

        let sign_bit_sum_builder: SignBitProductBuilder<F> = SignBitProductBuilder::new(pos_sign_bits, neg_sign_bits, res_positive, res_negative);
        let final_res = layers.add_gkr(sign_bit_sum_builder);

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![final_res.get_enum()],
            input_layers: vec![input_layer]
        };

        witness
    }
}
impl<F: FieldExt> PathCheckCircuit<F> {
    /// Constructor
    pub fn new(
        decision_node_paths_mle: DenseMle<F, DecisionNode<F>>, 
        leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
        bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
        num_copy: usize,
    ) -> Self {
        Self {
            decision_node_paths_mle,
            leaf_node_paths_mle,
            bin_decomp_diff_mle,
            num_copy,
        }
    }
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
        let num_copy = self.batched_decision_node_paths_mle.len();
        let num_copy_bits = log2(num_copy) as usize;
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        let mut combined_decision = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(self.batched_decision_node_paths_mle.clone());
        let mut combined_leaf = DenseMle::<F, LeafNode<F>>::combine_mle_batch(self.batched_leaf_node_paths_mle.clone());
        let mut combined_bit = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.batched_bin_decomp_diff_mle.clone());

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_decision), Box::new(&mut combined_leaf), Box::new(&mut combined_bit)];
        let input_layer_builder = InputLayerBuilder::<F>::new(input_mles, None, LayerId::Input(0));
        let input_layer: InputLayerEnum<F, Self::Transcript> = input_layer_builder.to_input_layer::<PublicInputLayer<F, _>>().to_enum();
        
        let pos_builders = self.batched_bin_decomp_diff_mle.iter_mut().map(
            |bin_decomp_mle| {
                bin_decomp_mle.add_prefix_bits(Some(
                    combined_bit.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                OneMinusSignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let pos_batched_builder = BatchedLayer::new(pos_builders);

        let neg_builders = self.batched_bin_decomp_diff_mle.iter_mut().map(
            |bin_decomp_mle| {
                bin_decomp_mle.add_prefix_bits(Some(
                    combined_bit.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                SignBit::new(bin_decomp_mle.clone())
            }
        ).collect_vec();
        let neg_batched_builder = BatchedLayer::new(neg_builders);

        let pos_sign_bits = layers.add_gkr(pos_batched_builder); // ID is 0
        let neg_sign_bits = layers.add_gkr(neg_batched_builder); // ID is 1

        let prev_node_left_builders = self.batched_decision_node_paths_mle.iter_mut().map(
            |dec_mle| {
                dec_mle.add_prefix_bits(Some(
                    combined_decision.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                PrevNodeLeftBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_left_batched_builder = BatchedLayer::new(prev_node_left_builders);

        let prev_node_right_builders = self.batched_decision_node_paths_mle.iter_mut().map(
            |dec_mle| {
                dec_mle.add_prefix_bits(Some(
                    combined_decision.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                PrevNodeRightBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let prev_right_batched_builder = BatchedLayer::new(prev_node_right_builders);

        let curr_node_decision_builders = self.batched_decision_node_paths_mle.iter_mut().map(
            |dec_mle| {
                dec_mle.add_prefix_bits(Some(
                    combined_decision.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                CurrNodeBuilderDecision::new(dec_mle.clone())
            }
        ).collect_vec();

        let curr_decision_batched_builder = BatchedLayer::new(curr_node_decision_builders);

        let curr_node_leaf_builders = self.batched_leaf_node_paths_mle.iter_mut().map(
            |leaf_mle| {
                leaf_mle.add_prefix_bits(Some(
                    combined_decision.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_copy_bits)
                    ).collect_vec()
                ));
                CurrNodeBuilderLeaf::new(leaf_mle.clone())
            }
        ).collect_vec();

        let curr_leaf_batched_builder = BatchedLayer::new(curr_node_leaf_builders);
    
        let curr_decision = layers.add_gkr(curr_decision_batched_builder); // ID is 2
        let curr_leaf = layers.add_gkr(curr_leaf_batched_builder); // ID is 3

        let curr_dec_leaf_builders = curr_decision.into_iter().zip(curr_leaf.into_iter()).map(
            |(dec_mle, leaf_mle)| {
                // let dec_mle_fix: DenseMle<F, F> = DenseMle::new_from_raw(dec_mle.mle_ref().bookkeeping_table, dec_mle.layer_id, None);
                // let leaf_mle_fix: DenseMle<F, F> = DenseMle::new_from_raw(leaf_mle.mle_ref().bookkeeping_table, leaf_mle.layer_id, None);
                ConcatBuilder::new(dec_mle, leaf_mle)
            }
        ).collect_vec();

        let curr_dec_leaf_batched_builder = BatchedLayer::new(curr_dec_leaf_builders);
        let curr_node_decision_leaf = layers.add_gkr(curr_dec_leaf_batched_builder); // ID is 4
        dbg!(&curr_node_decision_leaf);

        let prev_node_right = layers.add_gkr(prev_right_batched_builder); // ID is 5
        let prev_node_left = layers.add_gkr(prev_left_batched_builder); // ID is 6

        // --- Okay so we need to unbatch by FIRST grabbing all of the decisions and leaves on both sides ---
        // --- Then flattening those first ---
        // --- Then merging with each other the batched versions ---
        // We have to do this because that's what the batched expression expects
        // Unfortunately we'll have to rewire the gates after this
        let flattened_curr = unbatch_mles(curr_node_decision_leaf);
        dbg!(&flattened_curr);

        // --- Debug ---
        let mut flattened_curr_ref_clone = flattened_curr.clone().mle_ref();
        flattened_curr_ref_clone.index_mle_indices(0);
        for (idx, chal) in vec![1, 2, 1, 2, 1].into_iter().enumerate() {
            flattened_curr_ref_clone.fix_variable(idx, F::from(chal));
            dbg!(&flattened_curr_ref_clone);
        }

        let flattened_prev_right = unbatch_mles(prev_node_right);
        let flattened_prev_left = unbatch_mles(prev_node_left);
        
        let nonzero_gates = create_wiring_from_num_bits(1 << (flattened_prev_left.num_iterated_vars() - num_copy_bits));

        let res_neg = layers.add_add_gate_batched(nonzero_gates.clone(), flattened_curr.clone().mle_ref(), flattened_prev_left.mle_ref(), num_copy_bits); // ID is 7
        let res_pos = layers.add_add_gate_batched(nonzero_gates, flattened_curr.clone().mle_ref(), flattened_prev_right.mle_ref(), num_copy_bits); // ID is 8

        // let neg_sign_bits_fix = neg_sign_bits.into_iter().map(
        //     |neg_sign_bit_mle| {
        //         let no_prefix_mle: DenseMle<F, F> = DenseMle::new_from_raw(neg_sign_bit_mle.mle_ref().bookkeeping_table, neg_sign_bit_mle.layer_id, None);
        //         no_prefix_mle
        //     }
        // ).collect_vec();

        // let pos_sign_bits_fix = pos_sign_bits.into_iter().map(
        //     |pos_sign_bit_mle| {
        //         let no_prefix_mle: DenseMle<F, F> = DenseMle::new_from_raw(pos_sign_bit_mle.mle_ref().bookkeeping_table, pos_sign_bit_mle.layer_id, None);
        //         no_prefix_mle
        //     }
        // ).collect_vec();

        let res_neg_unflat = unflatten_mle(res_neg, num_copy_bits);
        let res_pos_unflat = unflatten_mle(res_pos, num_copy_bits);

        //dbg!(&neg_sign_bits_fix, &res_neg_unflat, &pos_sign_bits_fix, &res_pos_unflat);

        let sign_bit_product_builders = (pos_sign_bits.into_iter().zip(neg_sign_bits.into_iter())).zip(res_pos_unflat.into_iter().zip(res_neg_unflat.into_iter())).map(
            |((pos_sign_bits, neg_sign_bits), (res_positive, res_negative))| {
                dbg!(&pos_sign_bits);
                dbg!(&pos_sign_bits.mle_ref().mle_indices());
                dbg!(&neg_sign_bits);
                dbg!(&res_positive);
                dbg!(&res_negative);
                SignBitProductBuilder::new(pos_sign_bits, neg_sign_bits, res_positive, res_negative)
            }
        ).collect_vec();

        let sign_bit_product_batched_builder = BatchedLayer::new(sign_bit_product_builders);

        let sign_product_res = layers.add_gkr(sign_bit_product_batched_builder); // ID is 9
        let final_res = combine_zero_mle_ref(sign_product_res); // ID is 10

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![final_res.get_enum()],
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
}

pub struct OneMinusCheckCircuit<F: FieldExt> {
    pub bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> GKRCircuit<F> for OneMinusCheckCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.bin_decomp_diff_mle)];
        let input_layer_builder = InputLayerBuilder::<F>::new(input_mles, None, LayerId::Input(0));
        let input_layer = input_layer_builder.to_input_layer::<PublicInputLayer<F, _>>().to_enum();

        let pos_sign_bit_builder = OneMinusSignBit::new(self.bin_decomp_diff_mle.clone());
        let pos_sign_bits = layers.add_gkr(pos_sign_bit_builder);

        let dumb = DumbBuilder::new(pos_sign_bits);
        let dumber = layers.add_gkr(dumb);


        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![dumber.get_enum()],
            input_layers: vec![input_layer]
        };

        
        witness
    }
}
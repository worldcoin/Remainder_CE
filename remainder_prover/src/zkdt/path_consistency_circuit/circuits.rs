use itertools::Itertools;
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef}, zkdt::{structs::{DecisionNode, LeafNode, BinDecomp16Bit}, builders::ConcatBuilder}, prover::{GKRCircuit, Witness, Layers, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}}, layer::{LayerId, empty_layer::EmptyLayer}, gate::gate::BinaryOperation};

use super::circuit_builders::{OneMinusSignBit, SignBit, PrevNodeLeftBuilderDecision, PrevNodeRightBuilderDecision, CurrNodeBuilderDecision, CurrNodeBuilderLeaf, SignBitProductBuilder};

/// Helper!
pub fn create_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    let mut gates = (0.. (size-1)).map(
        |idx| (idx, 2*(idx + 1), idx)
    ).collect_vec();
    gates.push((size-1, 1, size-1));

    gates
}

/// Helper!
pub fn decision_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    (0 .. (size-1)).map(
        |idx| (idx, idx + 1, idx)
    ).collect_vec()
}

/// Helper!
pub fn leaf_add_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size-1, 0)]
}


/// Helper!
pub fn decision_mul_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    (0 .. (size-1)).map(
        |idx| (idx, idx, idx)
    ).collect_vec()
}

/// Helper!
pub fn leaf_mul_wiring_from_size(size: usize) -> Vec<(usize, usize, usize)> {
    vec![(0, size-1, 0)]
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

        let curr_node_decision_leaf_builder = ConcatBuilder::new(curr_decision, curr_leaf);
        let curr_node_decision_leaf_mle_ref = layers.add_gkr(curr_node_decision_leaf_builder).mle_ref();
        let prev_node_right_mle_ref = layers.add_gkr(prev_node_right_builder).mle_ref();
        let prev_node_left_mle_ref = layers.add_gkr(prev_node_left_builder).mle_ref();

        let nonzero_gates = create_wiring_from_size(1 << (prev_node_left_mle_ref.num_vars() - self.num_copy));

        let res_negative = layers.add_gate(nonzero_gates.clone(), curr_node_decision_leaf_mle_ref.clone(), prev_node_left_mle_ref, None, BinaryOperation::Add);
        let res_positive = layers.add_gate(nonzero_gates, curr_node_decision_leaf_mle_ref, prev_node_right_mle_ref, None, BinaryOperation::Add);

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


/// For checking whether a path is consistent (i.e. all children of
/// parent nodes is either 2i + 1 or 2i + 2, where `i` is the parent's
/// node ID) but with MULGATES!!!!!!!!!!!!!!!!!!!!!!!!!!
pub struct PathMulCheckCircuit<F: FieldExt> {
    decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
    leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
    bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
}

impl<F: FieldExt> GKRCircuit<F> for PathMulCheckCircuit<F> {
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

        let curr_decision_mle_ref = layers.add_gkr(curr_node_decision_builder).mle_ref();
        let curr_leaf_mle_ref = layers.add::<_, EmptyLayer<F, Self::Transcript>>(curr_node_leaf_builder).mle_ref();
        let prev_node_right_mle_ref = layers.add_gkr(prev_node_right_builder).mle_ref();
        let prev_node_left_mle_ref = layers.add_gkr(prev_node_left_builder).mle_ref();

        let nonzero_gates_add_decision = decision_add_wiring_from_size(1 << (prev_node_left_mle_ref.num_vars()));
        let nonzero_gates_add_leaf = leaf_add_wiring_from_size(1 << (prev_node_left_mle_ref.num_vars()));

        let res_negative_dec = layers.add_gate(nonzero_gates_add_decision.clone(), curr_decision_mle_ref.clone(), prev_node_left_mle_ref.clone(), None, BinaryOperation::Add);
        let res_positive_dec = layers.add_gate(nonzero_gates_add_decision, curr_decision_mle_ref, prev_node_right_mle_ref.clone(), None, BinaryOperation::Add);

        let res_negative_leaf = layers.add_gate(nonzero_gates_add_leaf.clone(), prev_node_left_mle_ref, curr_leaf_mle_ref.clone(), None, BinaryOperation::Add);
        let res_positive_leaf = layers.add_gate(nonzero_gates_add_leaf, prev_node_right_mle_ref, curr_leaf_mle_ref, None, BinaryOperation::Add);


        let nonzero_gates_mul_decision = decision_mul_wiring_from_size(1 << (pos_sign_bits.num_iterated_vars()));
        let nonzero_gates_mul_leaf = leaf_mul_wiring_from_size(1 << (pos_sign_bits.num_iterated_vars()));

        let dec_pos_prod = layers.add_gate(nonzero_gates_mul_decision.clone(), pos_sign_bits.mle_ref(), res_positive_dec.mle_ref(), None, BinaryOperation::Mul);
        let dec_neg_prod = layers.add_gate(nonzero_gates_mul_decision, neg_sign_bits.mle_ref(), res_negative_dec.mle_ref(), None, BinaryOperation::Mul);
        let leaf_pos_prod = layers.add_gate(nonzero_gates_mul_leaf.clone(), pos_sign_bits.mle_ref(), res_positive_leaf.mle_ref(), None, BinaryOperation::Mul);
        let leaf_neg_prod = layers.add_gate(nonzero_gates_mul_leaf, neg_sign_bits.mle_ref(), res_negative_leaf.mle_ref(), None, BinaryOperation::Mul);

        let witness: Witness<F, Self::Transcript> = Witness {
            layers,
            output_layers: vec![dec_pos_prod.mle_ref().get_enum(), dec_neg_prod.mle_ref().get_enum(), leaf_pos_prod.mle_ref().get_enum(), leaf_neg_prod.mle_ref().get_enum()],
            input_layers: vec![input_layer]
        };

        witness
    }
}
impl<F: FieldExt> PathMulCheckCircuit<F> {
    /// Constructor
    pub fn new(
        decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
        leaf_node_paths_mle: DenseMle<F, LeafNode<F>>,
        bin_decomp_diff_mle: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            decision_node_paths_mle,
            leaf_node_paths_mle,
            bin_decomp_diff_mle,
        }
    }
}
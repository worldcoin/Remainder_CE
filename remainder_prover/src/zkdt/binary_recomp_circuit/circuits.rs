use ark_std::log2;
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef}, zkdt::structs::{DecisionNode, InputAttribute, BinDecomp16Bit}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, ligero_input_layer::LigeroInputLayer, InputLayer}, Layers}, layer::LayerId};

use super::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder};

pub struct BinaryRecompCircuitBatched<F: FieldExt> {
    batched_decision_node_path_mle: Vec<DenseMle<F, DecisionNode<F>>>,
    batched_permuted_inputs_mle: Vec<DenseMle<F, InputAttribute<F>>>,
    batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuitBatched<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- For the input layer, we need to first merge all of the input MLEs FIRST by mle_idx, then by batched index ---
        // --- This assures that (going left-to-right in terms of the bits) we have [input_prefix_bits] [batched_bits], [mle_idx], [iterated_bits] ---
        let mut combined_batched_decision_node_path_mle = DenseMle::<F, DecisionNode<F>>::combine_mle_batch(self.batched_decision_node_path_mle.clone());
        let mut combined_batched_permuted_inputs_mle = DenseMle::<F, InputAttribute<F>>::combine_mle_batch(self.batched_permuted_inputs_mle.clone());
        let mut combined_batched_diff_signed_bin_decomp_mle = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.batched_diff_signed_bin_decomp_mle.clone());

        // --- Inputs to the circuit are just these three MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_batched_decision_node_path_mle), Box::new(&mut combined_batched_permuted_inputs_mle), Box::new(&mut combined_batched_diff_signed_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        // let input_layer_prefix_bits = input_layer_builder.fetch_prefix_bits();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_decision_node_path_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;
        debug_assert_eq!(num_dataparallel_bits, log2(self.batched_permuted_inputs_mle.len()) as usize);
        debug_assert_eq!(num_dataparallel_bits, log2(self.batched_diff_signed_bin_decomp_mle.len()) as usize);

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp builder ---
        let pos_bin_recomp_builders = (0..num_subcircuit_copies).map(|idx| {
            let diff_signed_bit_decomp_mle = self.batched_diff_signed_bin_decomp_mle[idx].clone();
            diff_signed_bit_decomp_mle.add_prefix_bits(combined_batched_diff_signed_bin_decomp_mle.prefix_bits);
            BinaryRecompBuilder::new(diff_signed_bit_decomp_mle)
        }).collect();

        
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);

        // --- Next, we create the diff ---
        // TODO!(ryancao): Combine this and the above layer!!!
        let diff_builder = NodePathDiffBuilder::new(
            self.decision_node_path_mle.clone(),
            self.permuted_inputs_mle.clone()
        );
        let raw_diff_mle = layers.add_gkr(diff_builder);

        // --- Finally, we create the checker ---
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            raw_diff_mle,
            self.diff_signed_bin_decomp.clone(),
            pos_bin_recomp_mle,
        );
        let recomp_checker_mle = layers.add_gkr(recomp_checker_builder);

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        Witness { layers, output_layers: vec![recomp_checker_mle.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}
impl<F: FieldExt> BinaryRecompCircuitBatched<F> {
    /// Creates a new instance of BinaryRecompCircuitBatched
    pub fn new(
        batched_decision_node_path_mle: Vec<DenseMle<F, DecisionNode<F>>>,
        batched_permuted_inputs_mle: Vec<DenseMle<F, InputAttribute<F>>>,
        batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            batched_decision_node_path_mle,
            batched_permuted_inputs_mle,
            batched_diff_signed_bin_decomp_mle,
        }
    }
}

/// Checks that the binary recomposition of the differences
/// \bar{x}.val - path_x.thr are computed correctly.
pub struct BinaryRecompCircuit<F: FieldExt> {
    decision_node_path_mle: DenseMle<F, DecisionNode<F>>,
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinaryRecompCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Inputs to the circuit are just these three MLEs ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.decision_node_path_mle), Box::new(&mut self.permuted_inputs_mle), Box::new(&mut self.diff_signed_bin_decomp)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let pos_bin_recomp_builder = BinaryRecompBuilder::new(self.diff_signed_bin_decomp.clone());
        let pos_bin_recomp_mle = layers.add_gkr(pos_bin_recomp_builder);

        // --- Next, we create the diff ---
        // TODO!(ryancao): Combine this and the above layer!!!
        let diff_builder = NodePathDiffBuilder::new(
            self.decision_node_path_mle.clone(),
            self.permuted_inputs_mle.clone()
        );
        let raw_diff_mle = layers.add_gkr(diff_builder);

        // --- Finally, we create the checker ---
        let recomp_checker_builder = BinaryRecompCheckerBuilder::new(
            raw_diff_mle,
            self.diff_signed_bin_decomp.clone(),
            pos_bin_recomp_mle,
        );
        let recomp_checker_mle = layers.add_gkr(recomp_checker_builder);

        // --- Create input layers ---
        let live_committed_input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        Witness { layers, output_layers: vec![recomp_checker_mle.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}
impl<F: FieldExt> BinaryRecompCircuit<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        decision_node_path_mle: DenseMle<F, DecisionNode<F>>,
        permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
        diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            decision_node_path_mle,
            permuted_inputs_mle,
            diff_signed_bin_decomp,
        }
    }
}

/// Checks that when we grab parts of an MleRef we still generate
/// correct claims (mostly for debugging purposes).
pub struct PartialBitsCheckerCircuit<F: FieldExt> {
    permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
    decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
    num_vars_to_grab: usize,
}
impl<F: FieldExt> GKRCircuit<F> for PartialBitsCheckerCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.permuted_inputs_mle), Box::new(&mut self.decision_node_paths_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));

        let mut layers = Layers::new();
        let builder = PartialBitsCheckerBuilder::new(self.permuted_inputs_mle.clone(), self.decision_node_paths_mle.clone(), self.num_vars_to_grab);
        let result = layers.add_gkr(builder);

        let input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        Witness { layers, output_layers: vec![result.get_enum()], input_layers: vec![input_layer.to_enum()] }
    }
}
impl<F: FieldExt> PartialBitsCheckerCircuit<F> {
    /// Creates a new instance of PartialBitsCheckerCircuit
    pub fn new(
        permuted_inputs_mle: DenseMle<F, InputAttribute<F>>,
        decision_node_paths_mle: DenseMle<F, DecisionNode<F>>,
        num_vars_to_grab: usize,
    ) -> Self {
        Self {
            permuted_inputs_mle,
            decision_node_paths_mle,
            num_vars_to_grab,
        }
    }
}
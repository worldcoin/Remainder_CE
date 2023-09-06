use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef}, zkdt::structs::{DecisionNode, InputAttribute, BinDecomp16Bit}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, ligero_input_layer::LigeroInputLayer, InputLayer}, Layers}, layer::LayerId};

use super::circuit_builders::{BinaryRecompBuilder, NodePathDiffBuilder, BinaryRecompCheckerBuilder, PartialBitsCheckerBuilder};

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
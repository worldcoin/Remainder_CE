use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{expression::generic_expr::Expression, layer::{LayerId, from_mle}, mle::{dense::DenseMle, Mle, MleRef, zero::ZeroMleRef}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, Layers}, zkdt::structs::{BinDecomp16Bit, BinDecomp4Bit}};

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0
pub struct BinDecomp16BitIsBinaryCircuit<F: FieldExt> {
    bin_decomp_16_bit_mle: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitIsBinaryCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one MLE ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.bin_decomp_16_bit_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let output_mle_ref = layers.add_gkr(from_mle(
            self.bin_decomp_16_bit_mle.clone(), 
            |diff_signed_bin_decomp_mle| {
                let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                dbg!(&combined_bin_decomp_mle_ref);
                Expression::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - Expression::Mle(combined_bin_decomp_mle_ref)
            }, 
            |_mle, id, prefix_bits| {
                ZeroMleRef::new(self.bin_decomp_16_bit_mle.num_iterated_vars(), prefix_bits, id)
            })
        );

        Witness { layers, output_layers: vec![output_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}
impl<F: FieldExt> BinDecomp16BitIsBinaryCircuit<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        bin_decomp_16_bit_mle: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            bin_decomp_16_bit_mle,
        }
    }
    /// Creates a `Witness` for the combined circuit without worrying about input layers
    pub fn yield_sub_circuit(&self) -> Witness<F, <BinDecomp16BitIsBinaryCircuit<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp16BitIsBinaryCircuit<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let output_mle_ref = layers.add_gkr(from_mle(
            self.bin_decomp_16_bit_mle.clone(), 
            |bin_decomp_16_bit_mle_mle| {
                let combined_bin_decomp_mle_ref = bin_decomp_16_bit_mle_mle.get_entire_mle_as_mle_ref();
                Expression::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - Expression::Mle(combined_bin_decomp_mle_ref)
            }, 
            |_mle, id, prefix_bits| {
                ZeroMleRef::new(self.bin_decomp_16_bit_mle.num_iterated_vars(), prefix_bits, id)
            })
        );
        println!("# layers -- bits r binary multiset: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![output_mle_ref.get_enum()], input_layers: vec![] }
    }
}

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0
pub struct BinDecomp4BitIsBinaryCircuit<F: FieldExt> {
    multiplicities_bin_decomp_mle_input: DenseMle<F, BinDecomp4Bit<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp4BitIsBinaryCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one MLE ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.multiplicities_bin_decomp_mle_input)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let output_mle_ref = layers.add_gkr(from_mle(
            self.multiplicities_bin_decomp_mle_input.clone(), 
            |diff_signed_bin_decomp_mle| {
                let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                dbg!(&combined_bin_decomp_mle_ref);
                Expression::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - Expression::Mle(combined_bin_decomp_mle_ref)
            }, 
            |_mle, id, prefix_bits| {
                ZeroMleRef::new(self.multiplicities_bin_decomp_mle_input.num_iterated_vars(), prefix_bits, id)
            })
        );

        Witness { layers, output_layers: vec![output_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}
impl<F: FieldExt> BinDecomp4BitIsBinaryCircuit<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        multiplicities_bin_decomp_mle_input: DenseMle<F, BinDecomp4Bit<F>>,
    ) -> Self {
        Self {
            multiplicities_bin_decomp_mle_input,
        }
    }
}
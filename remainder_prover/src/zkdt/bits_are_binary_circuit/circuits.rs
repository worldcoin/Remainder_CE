use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{mle::{dense::DenseMle, Mle, MleRef, zero::ZeroMleRef, MleIndex}, zkdt::structs::BinDecomp16Bit, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, Layers}, layer::{LayerId, batched::{BatchedLayer, combine_zero_mle_ref}, from_mle}, expression::ExpressionStandard};

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp16BitIsBinaryCircuitBatched<F: FieldExt> {
    batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitIsBinaryCircuitBatched<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one (combined) MLE ---
        let mut combined_batched_diff_signed_bin_decomp_mle = DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(self.batched_diff_signed_bin_decomp_mle.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_batched_diff_signed_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_diff_signed_bin_decomp_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
            diff_signed_bin_decomp_mle.add_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        combined_batched_diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                            repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                        )
                    ).collect_vec()
                )
            );
            let diff_builder = from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            });
            diff_builder
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}

impl<F: FieldExt> BinDecomp16BitIsBinaryCircuitBatched<F> {
    /// Creates a new instance of BinDecomp16BitIsBinaryCircuitBatched
    pub fn new(
        batched_diff_signed_bin_decomp_mle: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            batched_diff_signed_bin_decomp_mle,
        }
    }
}

/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0
pub struct BinDecomp16BitIsBinaryCircuit<F: FieldExt> {
    diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitIsBinaryCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one MLE ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut self.diff_signed_bin_decomp)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- First we create the positive binary recomp ---
        let output_mle_ref = layers.add_gkr(from_mle(
            self.diff_signed_bin_decomp.clone(), 
            |diff_signed_bin_decomp_mle| {
                let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                dbg!(&combined_bin_decomp_mle_ref);
                ExpressionStandard::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionStandard::Mle(combined_bin_decomp_mle_ref)
            }, 
            |_mle, id, prefix_bits| {
                ZeroMleRef::new(self.diff_signed_bin_decomp.num_iterated_vars(), prefix_bits, id)
            })
        );

        Witness { layers, output_layers: vec![output_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}
impl<F: FieldExt> BinDecomp16BitIsBinaryCircuit<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        diff_signed_bin_decomp: DenseMle<F, BinDecomp16Bit<F>>,
    ) -> Self {
        Self {
            diff_signed_bin_decomp,
        }
    }
}
use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{expression::generic_expr::Expression, layer::{LayerId, batched::{BatchedLayer, combine_zero_mle_ref}, from_mle}, mle::{dense::DenseMle, Mle, MleRef, zero::ZeroMleRef, MleIndex}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, Layers}, zkdt::structs::{BinDecomp16Bit, BinDecomp4Bit, BinDecomp8Bit}};

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
            diff_signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        combined_batched_diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                            repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                        )
                    ).collect_vec()
                )
            );
            
            from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    Expression::pow(2, combined_bin_decomp_mle_ref.clone()) - Expression::mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
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

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp16BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp16BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.batched_diff_signed_bin_decomp_mle.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
            diff_signed_bin_decomp_mle.set_prefix_bits(
                Some(
                    diff_signed_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            
            from_mle(
                diff_signed_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    Expression::pow(2, combined_bin_decomp_mle_ref.clone()) - Expression::mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));
        println!("# layers -- bits r binary 16bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }
}

/// Checks that all of the bits within a `BinDecomp4Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp4BitIsBinaryCircuitBatched<F: FieldExt> {
    multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp4Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp4BitIsBinaryCircuitBatched<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

        // --- Input to the circuit is just the one (combined) MLE ---
        let mut combined_batched_multiplicities_bin_decomp_mle = DenseMle::<F, BinDecomp4Bit<F>>::combine_mle_batch(self.multiplicities_bin_decomp_mle_input_vec.clone());
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut combined_batched_multiplicities_bin_decomp_mle)];
        let input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let live_committed_input_layer: PublicInputLayer<F, Self::Transcript> = input_layer_builder.to_input_layer();

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, Self::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.multiplicities_bin_decomp_mle_input_vec.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.multiplicities_bin_decomp_mle_input_vec.iter_mut().map(|multiplicities_bin_decomp_mle| {
            multiplicities_bin_decomp_mle.set_prefix_bits(
                Some(
                    multiplicities_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        combined_batched_multiplicities_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                            repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                        )
                    ).collect_vec()
                )
            );
            
            from_mle(
                multiplicities_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    Expression::pow(2, combined_bin_decomp_mle_ref.clone()) - Expression::mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![live_committed_input_layer.to_enum()] }
    }
}

impl<F: FieldExt> BinDecomp4BitIsBinaryCircuitBatched<F> {
    /// Creates a new instance of BinDecomp16BitIsBinaryCircuitBatched
    pub fn new(
        multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp4Bit<F>>>,
    ) -> Self {
        Self {
            multiplicities_bin_decomp_mle_input_vec,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp4BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp4BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.multiplicities_bin_decomp_mle_input_vec.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.multiplicities_bin_decomp_mle_input_vec.iter_mut().map(|multiplicities_bin_decomp_mle| {
            multiplicities_bin_decomp_mle.set_prefix_bits(
                Some(
                    multiplicities_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            
            from_mle(
                multiplicities_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    Expression::pow(2, combined_bin_decomp_mle_ref.clone()) - Expression::mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        println!("# layers -- bits r binary 4bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }   
}



/// Checks that all of the bits within a `BinDecomp8Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp8BitIsBinaryCircuitBatched<F: FieldExt> {
    multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp8BitIsBinaryCircuitBatched<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }
}

impl<F: FieldExt> BinDecomp8BitIsBinaryCircuitBatched<F> {
    /// Creates a new instance of BinDecomp16BitIsBinaryCircuitBatched
    pub fn new(
        multiplicities_bin_decomp_mle_input_vec: Vec<DenseMle<F, BinDecomp8Bit<F>>>,
    ) -> Self {
        Self {
            multiplicities_bin_decomp_mle_input_vec,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp8BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp4BitIsBinaryCircuitBatched<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_subcircuit_copies = self.multiplicities_bin_decomp_mle_input_vec.len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        // --- Create the builders for (b_i)^2 - b_i ---
        let diff_builders = self.multiplicities_bin_decomp_mle_input_vec.iter_mut().map(|multiplicities_bin_decomp_mle| {
            multiplicities_bin_decomp_mle.set_prefix_bits(
                Some(
                    multiplicities_bin_decomp_mle.get_prefix_bits().iter().flatten().cloned().chain(
                        repeat_n(MleIndex::Iterated, num_dataparallel_bits)
                    ).collect_vec()
                )
            );
            
            from_mle(
                multiplicities_bin_decomp_mle, 
                |diff_signed_bin_decomp_mle| {
                    let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                    Expression::pow(2, combined_bin_decomp_mle_ref.clone()) - Expression::mle(combined_bin_decomp_mle_ref)
                }, 
                |mle, id, prefix_bits| {
                    ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
            })
        }).collect_vec();
        let combined_output_zero_mle_ref = combine_zero_mle_ref(layers.add_gkr(BatchedLayer::new(diff_builders)));

        println!("# layers -- bits r binary 8bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }   
}
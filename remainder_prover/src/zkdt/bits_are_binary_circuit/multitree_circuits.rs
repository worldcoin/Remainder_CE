use ark_std::log2;
use itertools::{Itertools, repeat_n};
use remainder_shared_types::{FieldExt, transcript::poseidon_transcript::PoseidonTranscript};

use crate::{expression::generic_expr::ExpressionNode, layer::{LayerId, batched::{BatchedLayer, combine_zero_mle_ref}, from_mle}, mle::{dense::DenseMle, Mle, MleRef, zero::ZeroMleRef, MleIndex}, prover::{GKRCircuit, Witness, input_layer::{combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer, InputLayer}, Layers}, zkdt::structs::{BinDecomp16Bit, BinDecomp4Bit}};


pub struct BinDecomp16BitIsBinaryCircuitMultiTree<F: FieldExt> {
    bin_decomp_16_bit_mle_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitIsBinaryCircuitMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }
}


impl<F: FieldExt> BinDecomp16BitIsBinaryCircuitMultiTree<F> {
    /// Creates a new instance of BinaryRecompCircuit
    pub fn new(
        bin_decomp_16_bit_mle_tree: Vec<DenseMle<F, BinDecomp16Bit<F>>>,
    ) -> Self {
        Self {
            bin_decomp_16_bit_mle_tree,
        }
    }
    /// Creates a `Witness` for the combined circuit without worrying about input layers
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp16BitIsBinaryCircuitMultiTree<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp16BitIsBinaryCircuitMultiTree<F> as GKRCircuit<F>>::Transcript> = Layers::new();
        let num_tree_bits = log2(self.bin_decomp_16_bit_mle_tree.len()) as usize;

        self.bin_decomp_16_bit_mle_tree.iter_mut().for_each(|mle| {
            mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_tree_bits)).collect_vec()));
        });

        // --- First we create the positive binary recomp ---
        let output_mle_ref_vec = layers.add_gkr(BatchedLayer::new(  
            self.bin_decomp_16_bit_mle_tree.iter_mut().map(
                |bin_decomp_16_bit_mle| {

                from_mle(
            bin_decomp_16_bit_mle.clone(), 
            |bin_decomp_16_bit_mle_mle| {
                let combined_bin_decomp_mle_ref = bin_decomp_16_bit_mle_mle.get_entire_mle_as_mle_ref();
                ExpressionNode::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionNode::Mle(combined_bin_decomp_mle_ref)
            }, 
            |_mle, id, prefix_bits| {
                ZeroMleRef::new(bin_decomp_16_bit_mle.num_iterated_vars(), prefix_bits, id)
            })
            }

        ).collect_vec()));
        println!("# layers -- bits r binary multiset: {:?}", layers.next_layer_id());

        let output_mle_ref = combine_zero_mle_ref(output_mle_ref_vec);

        Witness { layers, output_layers: vec![output_mle_ref.get_enum()], input_layers: vec![] }
    }
}





/// Checks that all of the bits within a `BinDecomp16Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F: FieldExt> {
    batched_diff_signed_bin_decomp_tree_mle: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }
}

impl<F: FieldExt> BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F> {
    /// Creates a new instance of BinDecomp16BitIsBinaryCircuitBatched
    pub fn new(
        batched_diff_signed_bin_decomp_tree_mle: Vec<Vec<DenseMle<F, BinDecomp16Bit<F>>>>,
    ) -> Self {
        Self {
            batched_diff_signed_bin_decomp_tree_mle,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp16BitIsBinaryCircuitBatchedMultiTree<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_tree_bits = log2(self.batched_diff_signed_bin_decomp_tree_mle.len()) as usize;
        let num_subcircuit_copies = self.batched_diff_signed_bin_decomp_tree_mle[0].len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;

        self.batched_diff_signed_bin_decomp_tree_mle.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)).collect_vec()));
            })
        });


        let diff_builders_vec = self.batched_diff_signed_bin_decomp_tree_mle.clone().into_iter().map(|mut batched_diff_signed_bin_decomp_mle| {
            // --- Create the builders for (b_i)^2 - b_i ---
            let diff_builders = batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
                
                from_mle(
                    diff_signed_bin_decomp_mle.clone(), 
                    |diff_signed_bin_decomp_mle| {
                        let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                        ExpressionNode::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionNode::Mle(combined_bin_decomp_mle_ref)
                    }, 
                    |mle, id, prefix_bits| {
                        ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
                })
            }).collect_vec();
            BatchedLayer::new(diff_builders)

        }).collect_vec();

        let res_zero_mle_vec = layers.add_gkr(BatchedLayer::new(diff_builders_vec));
        
        let combined_output_zero_mle_ref = combine_zero_mle_ref(res_zero_mle_vec.into_iter().map(
            |res_zero_mle| combine_zero_mle_ref(res_zero_mle)
        ).collect_vec());

        println!("# layers -- bits r binary 16bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }
}

/// Checks that all of the bits within a `BinDecomp4Bit` are indeed binary
/// via b_i^2 - b_i = 0 (but it's batched)
pub struct BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F: FieldExt> {
    multiplicities_bin_decomp_mle_input_tree_vec: Vec<Vec<DenseMle<F, BinDecomp4Bit<F>>>>,
}
impl<F: FieldExt> GKRCircuit<F> for BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }
}

impl<F: FieldExt> BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F> {
    /// Creates a new instance of BinDecomp16BitIsBinaryCircuitBatched
    pub fn new(
        multiplicities_bin_decomp_mle_input_tree_vec: Vec<Vec<DenseMle<F, BinDecomp4Bit<F>>>>,
    ) -> Self {
        Self {
            multiplicities_bin_decomp_mle_input_tree_vec,
        }
    }

    /// This does exactly the same thing as `synthesize()` above, but
    /// takes in prefix bits for each of the input layer MLEs as opposed
    /// to synthesizing its own input layer.
    pub fn yield_sub_circuit(&mut self) -> Witness<F, <BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F> as GKRCircuit<F>>::Transcript> {

        // --- Create `Layers` struct to add layers to ---
        let mut layers: Layers<F, <BinDecomp4BitIsBinaryCircuitBatchedMultiTree<F> as GKRCircuit<F>>::Transcript> = Layers::new();

        // --- Dataparallel/batching stuff + sanitychecks ---
        let num_tree_bits = log2(self.multiplicities_bin_decomp_mle_input_tree_vec.len()) as usize;
        let num_subcircuit_copies = self.multiplicities_bin_decomp_mle_input_tree_vec[0].len();
        let num_dataparallel_bits = log2(num_subcircuit_copies) as usize;
        self.multiplicities_bin_decomp_mle_input_tree_vec.iter_mut().for_each(|mle_vec| {
            mle_vec.iter_mut().for_each(|mle| {
                mle.set_prefix_bits(Some(mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, num_dataparallel_bits + num_tree_bits)).collect_vec()));
            })
        });

        let diff_builders_vec = self.multiplicities_bin_decomp_mle_input_tree_vec.clone().into_iter().map(|mut batched_diff_signed_bin_decomp_mle| {
            // --- Create the builders for (b_i)^2 - b_i ---
            let diff_builders = batched_diff_signed_bin_decomp_mle.iter_mut().map(|diff_signed_bin_decomp_mle| {
                
                from_mle(
                    diff_signed_bin_decomp_mle.clone(), 
                    |diff_signed_bin_decomp_mle| {
                        let combined_bin_decomp_mle_ref = diff_signed_bin_decomp_mle.get_entire_mle_as_mle_ref();
                        ExpressionNode::Product(vec![combined_bin_decomp_mle_ref.clone(), combined_bin_decomp_mle_ref.clone()]) - ExpressionNode::Mle(combined_bin_decomp_mle_ref)
                    }, 
                    |mle, id, prefix_bits| {
                        ZeroMleRef::new(mle.num_iterated_vars(), prefix_bits, id)
                })
            }).collect_vec();
            BatchedLayer::new(diff_builders)

        }).collect_vec();

        let res_zero_mle_vec = layers.add_gkr(BatchedLayer::new(diff_builders_vec));
        
        let combined_output_zero_mle_ref = combine_zero_mle_ref(res_zero_mle_vec.into_iter().map(
            |res_zero_mle| combine_zero_mle_ref(res_zero_mle)
        ).collect_vec());

        println!("# layers -- bits r binary 4bit: {:?}", layers.next_layer_id());

        Witness { layers, output_layers: vec![combined_output_zero_mle_ref.get_enum()], input_layers: vec![] }
        
    }   
}
use ark_std::log2;
use itertools::{repeat_n, Itertools};

use crate::prover::{GKRCircuit, Layers, Witness};
use crate::{
    layer::{
        batched::{combine_zero_mle_ref, unbatch_mles, BatchedLayer},
        empty_layer::EmptyLayer,
        LayerBuilder, LayerId, Padding,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        combine_layers::combine_layers,
        input_layer::{
            self,
            combine_input_layers::InputLayerBuilder,
            enum_input_layer::{CommitmentEnum, InputLayerEnum},
            ligero_input_layer::LigeroInputLayer,
            random_input_layer::RandomInputLayer,
            InputLayer, MleInputLayer,
        },
        GKRError,
    },
    zkdt::builders::{AttributeConsistencyBuilderZeroRef, BitExponentiationBuilderCatBoost},
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonTranscript, Transcript},
    FieldExt,
};

use super::{
    attribute_consistency_circuit::dataparallel_circuits::AttributeConsistencyCircuit,
    binary_recomp_circuit::dataparallel_circuits::BinaryRecompCircuitBatched,
    bits_are_binary_circuit::{
        circuits::BinDecomp16BitIsBinaryCircuit,
        dataparallel_circuits::{
            BinDecomp16BitIsBinaryCircuitBatched, BinDecomp4BitIsBinaryCircuitBatched,
        },
    },
    builders::{
        BitExponentiationBuilderInput, EqualityCheck, FSDecisionPackingBuilder,
        FSInputPackingBuilder, FSLeafPackingBuilder, FSRMinusXBuilder, ProductBuilder,
        SplitProductBuilder, SquaringBuilder,
    },
    data_pipeline::{
        dt2zkdt::load_upshot_data_single_tree_batch, dummy_data_generator::BatchedCatboostMles,
    },
    input_multiset_circuit::dataparallel_circuits::InputMultiSetCircuit,
    multiset_circuit::{circuits::FSMultiSetCircuit, legacy_circuits::MultiSetCircuit},
    path_consistency_circuit::circuits::PathCheckCircuitBatchedMul,
    structs::{BinDecomp16Bit, BinDecomp4Bit, DecisionNode, InputAttribute, LeafNode},
};
use std::marker::PhantomData;

/// The actual ZKDT circuit!
pub struct CombinedCircuits<F: FieldExt> {
    /// All of the input MLEs coming from the data generation pipeline
    pub batched_catboost_mles: BatchedCatboostMles<F>,
}

impl<F: FieldExt> GKRCircuit<F> for CombinedCircuits<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        unimplemented!()
    }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Witness<F, Self::Transcript>,
            Vec<input_layer::enum_input_layer::CommitmentEnum<F>>,
        ),
        crate::prover::GKRError,
    > {
        let (
            // --- Actual circuit components ---
            mut attribute_consistency_circuit,
            mut multiset_circuit,
            mut input_multiset_circuit,
            mut binary_recomp_circuit_batched,
            mut path_consistency_circuit_batched,
            mut bin_decomp_4_bit_batched_bits_binary,
            mut bin_decomp_16_bit_batched_bits_binary,
            bits_are_binary_multiset_decision_circuit,
            bits_are_binary_multiset_leaf_circuit,
            // --- Input layer ---
            input_layers,
            inpunt_layers_commits,
        ) = self.create_sub_circuits(transcript).unwrap();

        let attribute_consistency_witness = attribute_consistency_circuit.yield_sub_circuit();
        let multiset_witness = multiset_circuit.yield_sub_circuit();
        let input_multiset_witness = input_multiset_circuit.yield_sub_circuit();
        let binary_recomp_circuit_batched_witness =
            binary_recomp_circuit_batched.yield_sub_circuit();
        let bin_decomp_4_bit_binary_batched_witness =
            bin_decomp_4_bit_batched_bits_binary.yield_sub_circuit();
        let bin_decomp_16_bit_binary_batched_witness =
            bin_decomp_16_bit_batched_bits_binary.yield_sub_circuit();
        let bits_are_binary_multiset_decision_circuit_witness =
            bits_are_binary_multiset_decision_circuit.yield_sub_circuit();
        let bits_are_binary_multiset_leaf_circuit_witness =
            bits_are_binary_multiset_leaf_circuit.yield_sub_circuit();

        let (mut combined_circuit_layers, combined_circuit_output_layers) = combine_layers(
            vec![
                attribute_consistency_witness.layers,
                multiset_witness.layers,
                input_multiset_witness.layers,
                binary_recomp_circuit_batched_witness.layers,
                bin_decomp_4_bit_binary_batched_witness.layers,
                bin_decomp_16_bit_binary_batched_witness.layers,
                bits_are_binary_multiset_decision_circuit_witness.layers,
                bits_are_binary_multiset_leaf_circuit_witness.layers,
            ],
            vec![
                attribute_consistency_witness.output_layers,
                multiset_witness.output_layers,
                input_multiset_witness.output_layers,
                binary_recomp_circuit_batched_witness.output_layers,
                bin_decomp_4_bit_binary_batched_witness.output_layers,
                bin_decomp_16_bit_binary_batched_witness.output_layers,
                bits_are_binary_multiset_decision_circuit_witness.output_layers,
                bits_are_binary_multiset_leaf_circuit_witness.output_layers,
            ],
        )
        .unwrap();

        // --- Manually add the layers and output layers from the circuit involving gate MLEs ---
        let updated_combined_output_layers = path_consistency_circuit_batched
            .add_subcircuit_layers_to_combined_layers(
                &mut combined_circuit_layers,
                combined_circuit_output_layers,
            );

        Ok((
            Witness {
                layers: combined_circuit_layers,
                output_layers: updated_combined_output_layers,
                input_layers,
            },
            inpunt_layers_commits,
        ))
    }
}

impl<F: FieldExt> CombinedCircuits<F> {
    fn create_sub_circuits(
        &mut self,
        transcript: &mut PoseidonTranscript<F>,
    ) -> Result<
        (
            AttributeConsistencyCircuit<F>,
            FSMultiSetCircuit<F>,
            InputMultiSetCircuit<F>,
            BinaryRecompCircuitBatched<F>,
            PathCheckCircuitBatchedMul<F>,
            BinDecomp4BitIsBinaryCircuitBatched<F>,
            BinDecomp16BitIsBinaryCircuitBatched<F>,
            BinDecomp16BitIsBinaryCircuit<F>,
            BinDecomp16BitIsBinaryCircuit<F>,
            Vec<InputLayerEnum<F, PoseidonTranscript<F>>>, // input layers, including random layers
            Vec<CommitmentEnum<F>>,                        // input layers' commitments
        ),
        GKRError,
    > {
        let BatchedCatboostMles {
            mut input_data_mle_vec,
            mut permuted_input_data_mle_vec,
            mut decision_node_paths_mle_vec,
            mut leaf_node_paths_mle_vec,
            mut multiplicities_bin_decomp_mle_decision,
            mut multiplicities_bin_decomp_mle_leaf,
            mut decision_nodes_mle,
            mut leaf_nodes_mle,
            mut binary_decomp_diffs_mle_vec,
            mut multiplicities_bin_decomp_mle_input_vec,
        } = self.batched_catboost_mles.clone(); // TODO!(% Labs): Get rid of this clone?!?!

        // deal w input
        let mut input_data_mle_combined =
            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(input_data_mle_vec.clone());
        let mut permuted_input_data_mle_vec_combined =
            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(
                permuted_input_data_mle_vec.clone(),
            );
        let mut decision_node_paths_mle_vec_combined =
            DenseMle::<F, DecisionNode<F>>::combine_mle_batch(decision_node_paths_mle_vec.clone());
        let mut leaf_node_paths_mle_vec_combined =
            DenseMle::<F, LeafNode<F>>::combine_mle_batch(leaf_node_paths_mle_vec.clone());
        let mut combined_batched_diff_signed_bin_decomp_mle =
            DenseMle::<F, BinDecomp16Bit<F>>::combine_mle_batch(
                binary_decomp_diffs_mle_vec.clone(),
            );
        let mut multiplicities_bin_decomp_mle_input_vec_combined =
            DenseMle::<F, BinDecomp4Bit<F>>::combine_mle_batch(
                multiplicities_bin_decomp_mle_input_vec.clone(),
            );

        // TODO!(ryancao): Actually fix this!!
        // Note to Veridise folks: The below should be split up as written, but we didn't quite have time to
        // fully debug it before sending things over. Will send an updated version ASAP!

        // Input layer shenanigans -- we need the following:
        // a) Precommitted Ligero input layer for tree itself (LayerId: 0)
        // b) Ligero input layer for just the inputs themselves (LayerId: 1)
        // c) Ligero input layer for all the auxiliaries (LayerId: 2)
        // d) Public input layer for all the leaf node outputs (TODO!(ryancao)): Actually do this bit! (LayerId: TODO!)
        // e) FS-style input layer for all the random packing constants + challenges (LayerId: 3)
        // let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
        //     Box::new(&mut decision_nodes_mle),
        //     Box::new(&mut leaf_nodes_mle),
        // ];
        // let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
        //     Box::new(&mut input_data_mle_combined),
        // ];
        // let aux_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
        //     Box::new(&mut permuted_input_data_mle_vec_combined),
        //     Box::new(&mut decision_node_paths_mle_vec_combined),
        //     Box::new(&mut leaf_node_paths_mle_vec_combined),
        //     Box::new(&mut multiplicities_bin_decomp_mle_decision),
        //     Box::new(&mut multiplicities_bin_decomp_mle_leaf),
        //     Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
        //     Box::new(&mut multiplicities_bin_decomp_mle_input_vec_combined),
        // ];

        // --- a) Precommitted Ligero input layer for tree itself (LayerId: 0) ---
        // let tree_mle_precommit_filepath = get_tree_commitment_filename_for_tree_number(0, Path::new("upshot_data/tree_ligero_commitments/"));
        // let (
        //     _ligero_encoding,
        //     ligero_commit,
        //     ligero_root,
        //     ligero_aux
        // ): (
        //     LigeroEncoding<F>,
        //     LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
        //     LcRoot<LigeroEncoding<F>, F>,
        //     LcProofAuxiliaryInfo,
        // ) = {
        //     let file = std::fs::File::open(tree_mle_precommit_filepath).unwrap();
        //     from_reader(&file).unwrap()
        // };
        // let tree_mle_input_layer_builder = InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));

        // --- b) Ligero input layer for just the inputs themselves (LayerId: 1) ---
        // let input_mles_input_layer_builder = InputLayerBuilder::new(input_mles, None, LayerId::Input(1));

        // --- c) Ligero input layer for all the auxiliaries (LayerId: 2) ---
        // let aux_mles_input_layer_builder = InputLayerBuilder::new(aux_mles, None, LayerId::Input(2));

        // --- Convert all the input layer builders into input layers ---
        // let tree_mle_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = tree_mle_input_layer_builder.to_input_layer_with_precommit(ligero_commit, ligero_aux, ligero_root);
        // let input_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = input_mles_input_layer_builder.to_input_layer();
        // let aux_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> = aux_mles_input_layer_builder.to_input_layer();
        // let mut tree_mle_input_layer = tree_mle_input_layer.to_enum();
        // let mut input_mles_input_layer = input_mles_input_layer.to_enum();
        // let mut aux_mles_input_layer = aux_mles_input_layer.to_enum();

        // --- Just have a single Ligero commitment for now ---
        let all_ligero_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            // --- Tree MLEs ---
            Box::new(&mut decision_nodes_mle),
            Box::new(&mut leaf_nodes_mle),
            // --- Input MLEs ---
            Box::new(&mut input_data_mle_combined),
            // --- Aux MLEs ---
            Box::new(&mut permuted_input_data_mle_vec_combined),
            Box::new(&mut decision_node_paths_mle_vec_combined),
            Box::new(&mut leaf_node_paths_mle_vec_combined),
            Box::new(&mut multiplicities_bin_decomp_mle_decision),
            Box::new(&mut multiplicities_bin_decomp_mle_leaf),
            Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
            Box::new(&mut multiplicities_bin_decomp_mle_input_vec_combined),
        ];
        let all_ligero_mles_input_layer_builder =
            InputLayerBuilder::new(all_ligero_mles, None, LayerId::Input(0));

        // --- Add input layer derived prefix bits to vectors ---
        // --- First input layer ---
        for decision_node_paths_mle in decision_node_paths_mle_vec.iter_mut() {
            decision_node_paths_mle
                .add_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits())
        }
        for leaf_node_paths_mle in leaf_node_paths_mle_vec.iter_mut() {
            leaf_node_paths_mle.add_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits())
        }

        // --- Second input layer ---
        for input_data_mle in input_data_mle_vec.iter_mut() {
            input_data_mle.add_prefix_bits(input_data_mle_combined.get_prefix_bits());
        }

        // --- Last input layer ---
        for permuted_input_data_mle in permuted_input_data_mle_vec.iter_mut() {
            permuted_input_data_mle
                .add_prefix_bits(permuted_input_data_mle_vec_combined.get_prefix_bits());
        }
        for decision_node_paths_mle in decision_node_paths_mle_vec.iter_mut() {
            decision_node_paths_mle
                .add_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits());
        }
        for leaf_node_paths_mle in leaf_node_paths_mle_vec.iter_mut() {
            leaf_node_paths_mle.add_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits());
        }
        for binary_decomp_diffs_mle in binary_decomp_diffs_mle_vec.iter_mut() {
            binary_decomp_diffs_mle
                .add_prefix_bits(combined_batched_diff_signed_bin_decomp_mle.get_prefix_bits())
        }
        for multiplicities_bin_decomp_mle_input in
            multiplicities_bin_decomp_mle_input_vec.iter_mut()
        {
            multiplicities_bin_decomp_mle_input
                .add_prefix_bits(multiplicities_bin_decomp_mle_input_vec_combined.get_prefix_bits())
        }

        // --- Add commitments to transcript so they are taken into account before the FS input layers are sampled ---
        // --- TODO!(ryancao): Do this correctly
        // let tree_mle_commit = tree_mle_input_layer
        //     .commit()
        //     .map_err(|err| GKRError::InputLayerError(err))?;
        // InputLayerEnum::append_commitment_to_transcript(&tree_mle_commit, transcript).unwrap();

        // let input_mle_commit = input_mles_input_layer
        //     .commit()
        //     .map_err(|err| GKRError::InputLayerError(err))?;
        // InputLayerEnum::append_commitment_to_transcript(&input_mle_commit, transcript).unwrap();

        // let aux_mle_commit = aux_mles_input_layer
        //     .commit()
        //     .map_err(|err| GKRError::InputLayerError(err))?;
        // InputLayerEnum::append_commitment_to_transcript(&aux_mle_commit, transcript).unwrap();

        let all_ligero_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            all_ligero_mles_input_layer_builder.to_input_layer();
        let mut all_ligero_mles_input_layer = all_ligero_mles_input_layer.to_enum();
        let all_input_mles_commit = all_ligero_mles_input_layer
            .commit()
            .map_err(GKRError::InputLayerError)?;
        InputLayerEnum::append_commitment_to_transcript(&all_input_mles_commit, transcript)
            .unwrap();

        // FS
        let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(1));
        let r_mle = random_r.get_mle();
        let mut random_r = random_r.to_enum();
        let random_r_commit = random_r.commit().map_err(GKRError::InputLayerError)?;

        let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(3));
        let r_packing_mle = random_r_packing.get_mle();
        let mut random_r_packing = random_r_packing.to_enum();
        let random_r_packing_commit = random_r_packing
            .commit()
            .map_err(GKRError::InputLayerError)?;

        let random_r_packing_another = RandomInputLayer::new(transcript, 1, LayerId::Input(4));
        let r_packing_another_mle = random_r_packing_another.get_mle();
        let mut random_r_packing_another = random_r_packing_another.to_enum();
        let random_r_packing_another_commit = random_r_packing_another
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // FS

        // construct the circuits

        let attribute_consistency_circuit = AttributeConsistencyCircuit {
            permuted_input_data_mle_vec: permuted_input_data_mle_vec.clone(),
            decision_node_paths_mle_vec: decision_node_paths_mle_vec.clone(),
        };

        // --- TDOO!(% Labs): Get rid of all the `.clone()`s ---
        let multiset_circuit = FSMultiSetCircuit {
            decision_nodes_mle,
            leaf_nodes_mle,
            multiplicities_bin_decomp_mle_decision: multiplicities_bin_decomp_mle_decision.clone(),
            multiplicities_bin_decomp_mle_leaf: multiplicities_bin_decomp_mle_leaf.clone(),
            decision_node_paths_mle_vec: decision_node_paths_mle_vec.clone(),
            leaf_node_paths_mle_vec: leaf_node_paths_mle_vec.clone(),
            r_mle: r_mle.clone(),
            r_packing_mle: r_packing_mle.clone(),
            r_packing_another_mle,
        };

        let input_multiset_circuit = InputMultiSetCircuit {
            input_data_mle_vec,
            permuted_input_data_mle_vec: permuted_input_data_mle_vec.clone(),
            multiplicities_bin_decomp_mle_input_vec: multiplicities_bin_decomp_mle_input_vec
                .clone(),
            r_mle,
            r_packing_mle,
        };

        let binary_recomp_circuit_batched = BinaryRecompCircuitBatched::new(
            decision_node_paths_mle_vec.clone(),
            permuted_input_data_mle_vec.clone(),
            binary_decomp_diffs_mle_vec.clone(),
        );

        let path_consistency_circuit_batched = PathCheckCircuitBatchedMul::new(
            decision_node_paths_mle_vec,
            leaf_node_paths_mle_vec,
            binary_decomp_diffs_mle_vec.clone(),
        );

        // --- Bits are binary check for each of the binary decomposition input MLEs ---
        let bits_binary_16_bit_batched =
            BinDecomp16BitIsBinaryCircuitBatched::new(binary_decomp_diffs_mle_vec);

        let bits_binary_4_bit_batched =
            BinDecomp4BitIsBinaryCircuitBatched::new(multiplicities_bin_decomp_mle_input_vec);

        let bits_are_binary_multiset_decision_circuit =
            BinDecomp16BitIsBinaryCircuit::new(multiplicities_bin_decomp_mle_decision);
        let bits_are_binary_multiset_leaf_circuit =
            BinDecomp16BitIsBinaryCircuit::new(multiplicities_bin_decomp_mle_leaf);

        Ok((
            // --- Actual circuit components ---
            attribute_consistency_circuit,
            multiset_circuit,
            input_multiset_circuit,
            binary_recomp_circuit_batched,
            path_consistency_circuit_batched,
            bits_binary_4_bit_batched,
            bits_binary_16_bit_batched,
            bits_are_binary_multiset_decision_circuit,
            bits_are_binary_multiset_leaf_circuit,
            // --- Input layers ---
            vec![
                // tree_mle_input_layer,
                // input_mles_input_layer,
                // aux_mles_input_layer,
                all_ligero_mles_input_layer,
                random_r,
                random_r_packing,
                random_r_packing_another,
            ],
            vec![
                // tree_mle_commit,
                // input_mle_commit,
                // aux_mle_commit,
                all_input_mles_commit,
                random_r_commit,
                random_r_packing_commit,
                random_r_packing_another_commit,
            ],
        ))
    }
}

/// GKRCircuit that proves inference for a single decision tree
pub struct ZKDTCircuit<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> ZKDTCircuit<F> {
    pub fn new(batch_size: usize) -> Self {
        let _batched_catboost_data =
            load_upshot_data_single_tree_batch::<F>(Some(batch_size), None);
        Self {
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt> GKRCircuit<F> for ZKDTCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::CombinedCircuits;
    use crate::prover::tests::test_circuit;
    use crate::zkdt::data_pipeline::dummy_data_generator::generate_mles_batch_catboost_single_tree;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use std::path::Path;

    use chrono;
    use env_logger::*;
    use log::{info, LevelFilter};
    use std::io::Write;

    #[test]
    fn test_combine_circuits() {
        env_logger::Builder::new()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "----> {}:{} {} [{}]:\n{}",
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
                    record.level(),
                    record.args()
                )
            })
            .filter(None, LevelFilter::Debug)
            .init();

        let (batched_catboost_mles, (_, _)) =
            generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let combined_circuit = CombinedCircuits {
            batched_catboost_mles,
        };

        test_circuit(combined_circuit, None);
    }
}

use ark_std::log2;
use itertools::{repeat_n, Itertools};
use remainder_ligero::{
    ligero_structs::LigeroEncoding, poseidon_ligero::PoseidonSpongeHasher, LcCommit,
    LcProofAuxiliaryInfo, LcRoot,
};
use serde_json::from_reader;

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
            public_input_layer::PublicInputLayer,
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

use ark_std::{end_timer, start_timer};

use super::input_data_to_circuit_adapter::BatchedZKDTCircuitMles;
use super::{
    attribute_consistency_circuit::dataparallel_circuits::AttributeConsistencyCircuit,
    binary_recomp_circuit::dataparallel_circuits::BinaryRecompCircuitBatched,
    bits_are_binary_circuit::{
        circuits::BinDecomp16BitIsBinaryCircuit,
        dataparallel_circuits::{
            BinDecomp16BitIsBinaryCircuitBatched, BinDecomp4BitIsBinaryCircuitBatched,
        },
    },
    input_multiset_circuit::dataparallel_circuits::InputMultiSetCircuit,
    multiset_circuit::circuits::FSMultiSetCircuit,
    path_consistency_circuit::circuits::PathCheckCircuitBatchedMul,
    structs::{BinDecomp16Bit, BinDecomp4Bit, DecisionNode, InputAttribute, LeafNode},
};

use std::{marker::PhantomData, path::Path};

/// The actual ZKDT circuit!
pub struct ZKDTCircuit<F: FieldExt> {
    /// All of the input MLEs coming from the data generation pipeline
    pub batched_zkdt_circuit_mles: BatchedZKDTCircuitMles<F>,
    /// The filepath to the precommitted tree that we are proving
    pub tree_precommit_filepath: String,
    /// The filepath to the precommitted sample minibatch that we are proving
    pub sample_minibatch_precommit_filepath: String,
}

impl<F: FieldExt> GKRCircuit<F> for ZKDTCircuit<F> {
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
        let create_sub_circuits_timer = start_timer!(|| "input + instantiate sub circuits");
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
        end_timer!(create_sub_circuits_timer);

        let wit_gen_timer = start_timer!(|| "witness generation of subcircuits");
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
        end_timer!(wit_gen_timer);

        let combine_layers_timer = start_timer!(|| "combine layers + gate stuff");
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

        end_timer!(combine_layers_timer);

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

impl<F: FieldExt> ZKDTCircuit<F> {
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
        let BatchedZKDTCircuitMles {
            mut input_samples_mle_vec,
            mut permuted_input_samples_mle_vec,
            mut decision_node_paths_mle_vec,
            mut leaf_node_paths_mle_vec,
            mut multiplicities_bin_decomp_mle_decision,
            mut multiplicities_bin_decomp_mle_leaf,
            mut decision_nodes_mle,
            mut leaf_nodes_mle,
            mut binary_decomp_diffs_mle_vec,
            mut multiplicities_bin_decomp_mle_input_vec,
        } = self.batched_zkdt_circuit_mles.clone(); // TODO!(% Labs): Get rid of this clone?!?!

        // deal w input
        let mut input_samples_mle_combined =
            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(input_samples_mle_vec.clone());
        let mut permuted_input_samples_mle_vec_combined =
            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(
                permuted_input_samples_mle_vec.clone(),
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

        // Input layer shenanigans -- we need the following:
        // a) Precommitted Ligero input layer for tree itself (LayerId: 0)
        // b) Ligero input layer for just the inputs themselves (LayerId: 1)
        // c) Ligero input layer for all the auxiliaries (LayerId: 2)
        // d) Public input layer for all the leaf node outputs (LayerId: 3)
        // e) FS-style input layer for all the random packing constants + challenges (LayerId: 4, 5, 6)
        // TODO!(ryancao): Make it so that we don't have to manually assign all of the layer IDs for input layer MLEs...

        // --- Input layer 0 ---
        decision_nodes_mle.layer_id = LayerId::Input(0);
        leaf_nodes_mle.layer_id = LayerId::Input(0);
        let tree_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut decision_nodes_mle),
            Box::new(&mut leaf_nodes_mle),
        ];

        // --- Input layer 1 ---
        input_samples_mle_combined.layer_id = LayerId::Input(1);
        input_samples_mle_vec.iter_mut().for_each(|mle| {
            mle.layer_id = LayerId::Input(1);
        });
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![Box::new(&mut input_samples_mle_combined)];

        // --- Input layer 2 ---
        permuted_input_samples_mle_vec_combined.layer_id = LayerId::Input(2);
        decision_node_paths_mle_vec_combined.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_decision.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_leaf.layer_id = LayerId::Input(2);
        combined_batched_diff_signed_bin_decomp_mle.layer_id = LayerId::Input(2);
        multiplicities_bin_decomp_mle_input_vec_combined.layer_id = LayerId::Input(2);
        permuted_input_samples_mle_vec.iter_mut().for_each(|mle| {
            mle.layer_id = LayerId::Input(2);
        });
        decision_node_paths_mle_vec.iter_mut().for_each(|mle| {
            mle.layer_id = LayerId::Input(2);
        });
        binary_decomp_diffs_mle_vec.iter_mut().for_each(|mle| {
            mle.layer_id = LayerId::Input(2);
        });
        multiplicities_bin_decomp_mle_input_vec
            .iter_mut()
            .for_each(|mle| {
                mle.layer_id = LayerId::Input(2);
            });
        let aux_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut permuted_input_samples_mle_vec_combined),
            Box::new(&mut decision_node_paths_mle_vec_combined),
            Box::new(&mut multiplicities_bin_decomp_mle_decision),
            Box::new(&mut multiplicities_bin_decomp_mle_leaf),
            Box::new(&mut combined_batched_diff_signed_bin_decomp_mle),
            Box::new(&mut multiplicities_bin_decomp_mle_input_vec_combined),
        ];

        // --- Input layer 3 ---
        leaf_node_paths_mle_vec_combined.layer_id = LayerId::Input(3);
        leaf_node_paths_mle_vec.iter_mut().for_each(|mle| {
            mle.layer_id = LayerId::Input(3);
        });
        let public_path_leaf_node_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut leaf_node_paths_mle_vec_combined)];

        // --- a) Precommitted Ligero input layer for tree itself (LayerId: 0) ---
        let tree_mle_input_layer_builder =
            InputLayerBuilder::new(tree_mles, None, LayerId::Input(0));

        // --- b) Ligero input layer for just the inputs themselves (LayerId: 1) ---
        let input_mles_input_layer_builder =
            InputLayerBuilder::new(input_mles, None, LayerId::Input(1));

        // --- c) Ligero input layer for all the auxiliaries (LayerId: 2) ---
        let aux_mles_input_layer_builder =
            InputLayerBuilder::new(aux_mles, None, LayerId::Input(2));

        // --- d) Public input layer for the path leaf nodes (LayerId: 3) ---
        let public_path_leaf_node_mles_input_layer_builder =
            InputLayerBuilder::new(public_path_leaf_node_mles, None, LayerId::Input(3));

        // --- Convert all the input layer builders into input layers ---
        let (_ligero_encoding, tree_ligero_commit, tree_ligero_root, tree_ligero_aux): (
            LigeroEncoding<F>,
            LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
            LcRoot<LigeroEncoding<F>, F>,
            LcProofAuxiliaryInfo,
        ) = {
            let file = std::fs::File::open(&self.tree_precommit_filepath).unwrap();
            from_reader(&file).unwrap()
        };
        let tree_mle_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            tree_mle_input_layer_builder.to_input_layer_with_precommit(
                tree_ligero_commit,
                tree_ligero_aux,
                tree_ligero_root,
            );

        let (
            _ligero_encoding,
            sample_minibatch_ligero_commit,
            sample_minibatch_ligero_root,
            sample_minibatch_ligero_aux,
        ): (
            LigeroEncoding<F>,
            LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
            LcRoot<LigeroEncoding<F>, F>,
            LcProofAuxiliaryInfo,
        ) = {
            let file = std::fs::File::open(&self.sample_minibatch_precommit_filepath).unwrap();
            from_reader(&file).unwrap()
        };
        let input_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            input_mles_input_layer_builder.to_input_layer_with_precommit(
                sample_minibatch_ligero_commit,
                sample_minibatch_ligero_aux,
                sample_minibatch_ligero_root,
            );

        let aux_mles_input_layer: LigeroInputLayer<F, PoseidonTranscript<F>> =
            aux_mles_input_layer_builder.to_input_layer();
        let public_path_leaf_node_mles_input_layer: PublicInputLayer<F, PoseidonTranscript<F>> =
            public_path_leaf_node_mles_input_layer_builder.to_input_layer();
        let mut tree_mle_input_layer = tree_mle_input_layer.to_enum();
        let mut input_mles_input_layer = input_mles_input_layer.to_enum();
        let mut aux_mles_input_layer = aux_mles_input_layer.to_enum();
        let mut public_path_leaf_node_mles_input_layer =
            public_path_leaf_node_mles_input_layer.to_enum();

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
        for input_samples_mle in input_samples_mle_vec.iter_mut() {
            input_samples_mle.add_prefix_bits(input_samples_mle_combined.get_prefix_bits());
        }

        // --- Third input layer ---
        for permuted_input_samples_mle in permuted_input_samples_mle_vec.iter_mut() {
            permuted_input_samples_mle
                .add_prefix_bits(permuted_input_samples_mle_vec_combined.get_prefix_bits());
        }
        for decision_node_paths_mle in decision_node_paths_mle_vec.iter_mut() {
            decision_node_paths_mle
                .add_prefix_bits(decision_node_paths_mle_vec_combined.get_prefix_bits());
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

        // --- Last input layer ---
        for leaf_node_paths_mle in leaf_node_paths_mle_vec.iter_mut() {
            leaf_node_paths_mle.add_prefix_bits(leaf_node_paths_mle_vec_combined.get_prefix_bits());
        }

        // --- Add commitments to transcript so they are taken into account before the FS input layers are sampled ---
        let tree_mle_commit = tree_mle_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&tree_mle_commit, transcript).unwrap();

        let input_mle_commit = input_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&input_mle_commit, transcript).unwrap();

        let aux_mle_commit = aux_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(&aux_mle_commit, transcript).unwrap();

        let public_path_leaf_node_mle_commit = public_path_leaf_node_mles_input_layer
            .commit()
            .map_err(|err| GKRError::InputLayerError(err))?;
        InputLayerEnum::append_commitment_to_transcript(
            &public_path_leaf_node_mle_commit,
            transcript,
        )
        .unwrap();

        // --- FS layers must also have LayerId::Input(.)s! ---
        // Input(4)
        let random_r = RandomInputLayer::new(transcript, 1, LayerId::Input(4));
        let r_mle = random_r.get_mle();
        let mut random_r = random_r.to_enum();
        let random_r_commit = random_r.commit().map_err(GKRError::InputLayerError)?;

        // Input(5)
        let random_r_packing = RandomInputLayer::new(transcript, 1, LayerId::Input(5));
        let r_packing_mle = random_r_packing.get_mle();
        let mut random_r_packing = random_r_packing.to_enum();
        let random_r_packing_commit = random_r_packing
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // Input(6)
        let random_r_packing_another = RandomInputLayer::new(transcript, 1, LayerId::Input(6));
        let r_packing_another_mle = random_r_packing_another.get_mle();
        let mut random_r_packing_another = random_r_packing_another.to_enum();
        let random_r_packing_another_commit = random_r_packing_another
            .commit()
            .map_err(GKRError::InputLayerError)?;

        // --- Construct the actual circuit structs ---
        let attribute_consistency_circuit = AttributeConsistencyCircuit {
            permuted_input_data_mle_vec: permuted_input_samples_mle_vec.clone(),
            decision_node_paths_mle_vec: decision_node_paths_mle_vec.clone(),
        };

        // --- TODO!(% Labs): Get rid of all the `.clone()`s ---
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
            input_data_mle_vec: input_samples_mle_vec,
            permuted_input_data_mle_vec: permuted_input_samples_mle_vec.clone(),
            multiplicities_bin_decomp_mle_input_vec: multiplicities_bin_decomp_mle_input_vec
                .clone(),
            r_mle,
            r_packing_mle,
        };

        let binary_recomp_circuit_batched = BinaryRecompCircuitBatched::new(
            decision_node_paths_mle_vec.clone(),
            permuted_input_samples_mle_vec.clone(),
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
                tree_mle_input_layer,
                input_mles_input_layer,
                aux_mles_input_layer,
                public_path_leaf_node_mles_input_layer,
                random_r,
                random_r_packing,
                random_r_packing_another,
            ],
            vec![
                tree_mle_commit,
                input_mle_commit,
                aux_mle_commit,
                public_path_leaf_node_mle_commit,
                random_r_commit,
                random_r_packing_commit,
                random_r_packing_another_commit,
            ],
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::ZKDTCircuit;
    use crate::prover::tests::test_circuit;
    use crate::zkdt::cache_upshot_catboost_inputs_for_testing::generate_mles_batch_catboost_single_tree;
    use ark_std::{end_timer, start_timer};
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use std::path::Path;

    use chrono;
    use env_logger::*;
    use log::{info, LevelFilter};
    use std::io::Write;

    #[test]
    fn test_zkdt_circuit() {
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
            .filter(None, LevelFilter::Error)
            .init();

        let (batched_catboost_mles, (_, _)) =
            generate_mles_batch_catboost_single_tree::<Fr>(1, Path::new("upshot_data/"));

        let combined_circuit = ZKDTCircuit {
            batched_zkdt_circuit_mles: batched_catboost_mles,
            tree_precommit_filepath: "upshot_data/tree_ligero_commitments/tree_commitment_0.json".to_string(),
            sample_minibatch_precommit_filepath: "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_10_commitment_0.json".to_string(),
        };

        test_circuit(
            combined_circuit,
            Some(Path::new("upshot_data/zkdt_proof_opt.json")),
        );
    }

    #[test]
    fn bench_zkdt_circuits() {
        (10..11).for_each(|batch_size| {
            let circuit_timer = start_timer!(|| format!("zkdt circuit, batch_size 2^{batch_size}"));
            let wit_gen_timer = start_timer!(|| "wit gen");

            let (batched_zkdt_circuit_mles, (_, _)) = generate_mles_batch_catboost_single_tree::<Fr>(
                batch_size,
                Path::new("upshot_data/"),
            );
            end_timer!(wit_gen_timer);

            let combined_circuit = ZKDTCircuit {
                batched_zkdt_circuit_mles,
                tree_precommit_filepath:
                    "upshot_data/tree_ligero_commitments/tree_commitment_0.json".to_string(),
                sample_minibatch_precommit_filepath:
                    "upshot_data/sample_minibatch_commitments/sample_minibatch_logsize_10_commitment_0.json".to_string(),
            };

            test_circuit(combined_circuit, None);

            end_timer!(circuit_timer);
        });
    }
}

use ark_std::{log2};
use itertools::{Itertools, repeat_n};
use ark_std::{start_timer, end_timer};

use crate::{mle::{dense::DenseMle, MleRef, Mle, MleIndex}, layer::{batched::{BatchedLayer, combine_zero_mle_ref}, LayerId}, zkdt::builders::{AttributeConsistencyBuilderZeroRef}, prover::{input_layer::{ligero_input_layer::LigeroInputLayer, combine_input_layers::InputLayerBuilder, InputLayer, MleInputLayer}}};
use crate::{prover::{GKRCircuit, Layers, Witness}};
use remainder_shared_types::{FieldExt, transcript::{Transcript, poseidon_transcript::PoseidonTranscript}};

use super::super::{structs::{InputAttribute, DecisionNode}};

/// Checks that the path nodes supplied by the prover have attributes which are
/// consistent against the attributes within \bar{x} by subtracting them.
pub(crate) struct AttributeConsistencyCircuitMultiTree<F: FieldExt> {
    pub permuted_input_data_mle_trees_vec: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
    pub decision_node_paths_mle_trees_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuitMultiTree<F> {
    type Transcript = PoseidonTranscript<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Transcript> {

          // input stuff
          let permuted_mle_combined_per_tree_vec = self.permuted_input_data_mle_trees_vec.clone().into_iter().map(
            |permuted_input_data_mle_vec| {
                DenseMle::<F, InputAttribute<F>>::combine_mle_batch(permuted_input_data_mle_vec.clone())
            }
          ).collect_vec();
          let mut permuted_tree_combined = DenseMle::<F, F>::combine_mle_batch(permuted_mle_combined_per_tree_vec);

          let decision_mle_combined_per_tree_vec = self.decision_node_paths_mle_trees_vec.clone().into_iter().map(
            |decision_node_paths_mle_vec| {
                DenseMle::<F, DecisionNode<F>>::combine_mle_batch(decision_node_paths_mle_vec.clone())
            }
          ).collect_vec();
          let mut decision_tree_combined = DenseMle::<F, F>::combine_mle_batch(decision_mle_combined_per_tree_vec);
            
          let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
              Box::new(&mut permuted_tree_combined),
              Box::new(&mut decision_tree_combined),
          ];
          let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
          let input_layer: LigeroInputLayer<F, Self::Transcript> = input_layer.to_input_layer_with_rho_inv(
            4,
            1_f64
          );
  
        let mut layers: Layers<_, Self::Transcript> = Layers::new();

        // rest of the stuff
        
        let tree_bits = log2(self.decision_node_paths_mle_trees_vec.len()) as usize;

        let attr_consistency_builder_per_tree = self.decision_node_paths_mle_trees_vec.clone().into_iter().zip(self.permuted_input_data_mle_trees_vec.clone()).map(|(decision_node_paths_mle_vec, permuted_input_data_mle_vec)| {
            let tree_height = (1 << (decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;
    
            // --- Number of dataparallel circuit copies ---
            let batch_bits = log2(decision_node_paths_mle_vec.len()) as usize;
    
            let attribute_consistency_builder = BatchedLayer::new(
    
                permuted_input_data_mle_vec
                        .iter()
                        .zip(decision_node_paths_mle_vec.iter())
                        .map(|(input_data_mle, decision_path_mle)| {
    
                            let mut input_data_mle = input_data_mle.clone();
                            input_data_mle.set_prefix_bits(Some(permuted_tree_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits + tree_bits)).collect_vec()));
    
                            let mut decision_path_mle = decision_path_mle.clone();
                            decision_path_mle.set_prefix_bits(Some(decision_tree_combined.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits + tree_bits)).collect_vec()));
    
                            // --- Simply subtracts the input data attribute IDs from the decision node attribute IDs ---
                            AttributeConsistencyBuilderZeroRef::new(
                                input_data_mle,
                                decision_path_mle,
                                tree_height
                            )
    
            }).collect_vec());

            attribute_consistency_builder

        }).collect_vec();

        let tree_batched_attribute_consistency_builder = BatchedLayer::new(attr_consistency_builder_per_tree);
       

        let difference_mle = layers.add_gkr(tree_batched_attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle.into_iter().map(
            |inner_zero_mle_ref| {
                combine_zero_mle_ref(inner_zero_mle_ref)
            }
        ).collect_vec());

        println!("# layers -- attr consis: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
        
            
    }
}

impl<F: FieldExt> AttributeConsistencyCircuitMultiTree<F> {
    pub fn new(
        permuted_input_data_mle_trees_vec: Vec<Vec<DenseMle<F, InputAttribute<F>>>>,
        decision_node_paths_mle_trees_vec: Vec<Vec<DenseMle<F, DecisionNode<F>>>>,
    ) -> Self {
        Self {
            permuted_input_data_mle_trees_vec,
            decision_node_paths_mle_trees_vec
        }
    }
    
    pub fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonTranscript<F>> {

        let mut layers: Layers<_, PoseidonTranscript<F>> = Layers::new();

        let tree_bits = log2(self.decision_node_paths_mle_trees_vec.len()) as usize;

        let attr_consistency_builder_per_tree = self.decision_node_paths_mle_trees_vec.clone().into_iter().zip(self.permuted_input_data_mle_trees_vec.clone()).map(|(decision_node_paths_mle_vec, permuted_input_data_mle_vec)| {
            let tree_height = (1 << (decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;
    
            // --- Number of dataparallel circuit copies ---
            let batch_bits = log2(decision_node_paths_mle_vec.len()) as usize;
    
            let attribute_consistency_builder = BatchedLayer::new(
    
                permuted_input_data_mle_vec
                        .iter()
                        .zip(decision_node_paths_mle_vec.iter())
                        .map(|(input_data_mle, decision_path_mle)| {
    
                            let mut input_data_mle = input_data_mle.clone();
                            input_data_mle.set_prefix_bits(Some(input_data_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits + tree_bits)).collect_vec()));
    
                            let mut decision_path_mle = decision_path_mle.clone();
                            decision_path_mle.set_prefix_bits(Some(decision_path_mle.get_prefix_bits().unwrap().into_iter().chain(repeat_n(MleIndex::Iterated, batch_bits + tree_bits)).collect_vec()));
    
                            // --- Simply subtracts the input data attribute IDs from the decision node attribute IDs ---
                            AttributeConsistencyBuilderZeroRef::new(
                                input_data_mle,
                                decision_path_mle,
                                tree_height
                            )
    
            }).collect_vec());

            attribute_consistency_builder

        }).collect_vec();

        let tree_batched_attribute_consistency_builder = BatchedLayer::new(attr_consistency_builder_per_tree);


       

        let difference_mle = layers.add_gkr(tree_batched_attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle.into_iter().map(
            |inner_zero_mle_ref| {
                combine_zero_mle_ref(inner_zero_mle_ref)
            }
        ).collect_vec());

        println!("# layers -- attr consis: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}
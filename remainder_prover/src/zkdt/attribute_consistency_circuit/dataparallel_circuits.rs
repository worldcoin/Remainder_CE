use ark_std::log2;
use ark_std::{end_timer, start_timer};
use itertools::{repeat_n, Itertools};

use crate::prover::{GKRCircuit, Layers, Witness};
use crate::{
    layer::{
        batched::{combine_zero_mle_ref, BatchedLayer},
        LayerId,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::input_layer::{
        combine_input_layers::InputLayerBuilder, ligero_input_layer::LigeroInputLayer, InputLayer,
        MleInputLayer,
    },
    zkdt::builders::AttributeConsistencyBuilderZeroRef,
};
use remainder_shared_types::{
    transcript::{poseidon_transcript::PoseidonSponge, Transcript},
    FieldExt,
};

use super::super::structs::{DecisionNode, InputAttribute};

/// Checks that the path nodes supplied by the prover have attributes which are
/// consistent against the attributes within \bar{x} by subtracting them.
pub(crate) struct AttributeConsistencyCircuit<F: FieldExt> {
    pub permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
    pub decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
}

impl<F: FieldExt> GKRCircuit<F> for AttributeConsistencyCircuit<F> {
    type Sponge = PoseidonSponge<F>;
    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;

        // --- Input layer combination shenanigans ---
        let mut dummy_permuted_input_data_mle_combined =
            DenseMle::<F, InputAttribute<F>>::combine_mle_batch(
                self.permuted_input_data_mle_vec.clone(),
            );
        let mut dummy_decision_node_paths_mle_combined =
            DenseMle::<F, DecisionNode<F>>::combine_mle_batch(
                self.decision_node_paths_mle_vec.clone(),
            );

        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut dummy_permuted_input_data_mle_combined),
            Box::new(&mut dummy_decision_node_paths_mle_combined),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0));
        let _input_prefix_bits = input_layer.fetch_prefix_bits();
        let input_layer: LigeroInputLayer<F, Self::Sponge> = input_layer.to_input_layer();

        let mut layers: Layers<_, Self::Sponge> = Layers::new();

        // --- Number of dataparallel circuit copies ---
        let batch_bits = log2(self.permuted_input_data_mle_vec.len()) as usize;

        let attribute_consistency_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec
                .iter()
                .zip(self.decision_node_paths_mle_vec.iter())
                .map(|(input_data_mle, decision_path_mle)| {
                    let mut input_data_mle = input_data_mle.clone();
                    input_data_mle.set_prefix_bits(Some(
                        dummy_permuted_input_data_mle_combined
                            .get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(MleIndex::Iterated, batch_bits))
                            .collect_vec(),
                    ));

                    let mut decision_path_mle = decision_path_mle.clone();
                    decision_path_mle.set_prefix_bits(Some(
                        dummy_decision_node_paths_mle_combined
                            .get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(MleIndex::Iterated, batch_bits))
                            .collect_vec(),
                    ));

                    // --- Simply subtracts the input data attribute IDs from the decision node attribute IDs ---
                    AttributeConsistencyBuilderZeroRef::new(
                        input_data_mle,
                        decision_path_mle,
                        tree_height,
                    )
                })
                .collect_vec(),
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle);

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![input_layer.to_enum()],
        }
    }
}

impl<F: FieldExt> AttributeConsistencyCircuit<F> {
    pub fn new(
        permuted_input_data_mle_vec: Vec<DenseMle<F, InputAttribute<F>>>,
        decision_node_paths_mle_vec: Vec<DenseMle<F, DecisionNode<F>>>,
    ) -> Self {
        Self {
            permuted_input_data_mle_vec,
            decision_node_paths_mle_vec,
        }
    }

    pub fn yield_sub_circuit(&mut self) -> Witness<F, PoseidonSponge<F>> {
        let tree_height = (1 << (self.decision_node_paths_mle_vec[0].num_iterated_vars() - 2)) + 1;
        let mut layers: Layers<_, PoseidonSponge<F>> = Layers::new();

        // --- Number of dataparallel circuit copies ---
        let batch_bits = log2(self.permuted_input_data_mle_vec.len()) as usize;

        let attribute_consistency_builder = BatchedLayer::new(
            self.permuted_input_data_mle_vec
                .iter()
                .zip(self.decision_node_paths_mle_vec.iter())
                .map(|(input_data_mle, decision_path_mle)| {
                    let mut input_data_mle = input_data_mle.clone();
                    input_data_mle.set_prefix_bits(Some(
                        input_data_mle
                            .get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(MleIndex::Iterated, batch_bits))
                            .collect_vec(),
                    ));

                    let mut decision_path_mle = decision_path_mle.clone();
                    decision_path_mle.set_prefix_bits(Some(
                        decision_path_mle
                            .get_prefix_bits()
                            .unwrap()
                            .into_iter()
                            .chain(repeat_n(MleIndex::Iterated, batch_bits))
                            .collect_vec(),
                    ));

                    // --- Simply subtracts the input data attribute IDs from the decision node attribute IDs ---
                    AttributeConsistencyBuilderZeroRef::new(
                        input_data_mle,
                        decision_path_mle,
                        tree_height,
                    )
                })
                .collect_vec(),
        );

        let difference_mle = layers.add_gkr(attribute_consistency_builder);
        let circuit_output = combine_zero_mle_ref(difference_mle);

        println!("# layers -- attr consis: {:?}", layers.next_layer_id());

        Witness {
            layers,
            output_layers: vec![circuit_output.get_enum()],
            input_layers: vec![],
        }
    }
}

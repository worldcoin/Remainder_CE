//! A module for defining how certain nodes can be Compiled into a GKR Witness

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::expression::circuit_expr::CircuitMle;
use crate::input_layer::enum_input_layer::{CircuitInputLayerEnum, InputLayerEnum};
use crate::input_layer::{CircuitInputLayer, InputLayer};
use crate::layer::layer_enum::{CircuitLayerEnum, LayerEnum};
use crate::layer::CircuitLayer;
use crate::layouter::layouting::CircuitMap;
use crate::layouter::nodes::circuit_inputs::compile_inputs::combine_input_mles;
use crate::layouter::nodes::CircuitNode;
use crate::mle::evals::MultilinearExtension;
use crate::output_layer::mle_output_layer::CircuitMleOutputLayer;
use crate::output_layer::CircuitOutputLayer;
use crate::prover::{GKRCircuitDescription, GKRError};
use crate::{
    layer::LayerId,
    layouter::layouting::layout,
    prover::{proof_system::DefaultProofSystem, GKRCircuit},
};
use itertools::Itertools;
use remainder_shared_types::transcript::ProverTranscript;
use remainder_shared_types::FieldExt;

use super::layouting::{CircuitDescriptionMap, CircuitLocation, InputLayerHintMap, InputNodeMap};
use super::nodes::circuit_inputs::InputLayerData;
use super::nodes::NodeId;
use super::{
    component::Component,
    nodes::{node_enum::NodeEnum, Context},
};

/// A basic circuit that uses the Layouter to construct the witness
pub struct LayouterCircuit<F: FieldExt, C: Component<NodeEnum<F>>, Fn: FnMut(&Context) -> C> {
    witness_builder: Fn,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, C: Component<NodeEnum<F>>, Fn: FnMut(&Context) -> C> LayouterCircuit<F, C, Fn> {
    /// Constructs a `LayouterCircuit` by taking in a closure that computes a Component
    /// that contains all the nodes that will be layedout and compiled into the witness
    pub fn new(witness_builder: Fn) -> Self {
        Self {
            witness_builder,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, C: Component<NodeEnum<F>>, Fn: FnMut(&Context) -> C> GKRCircuit<F>
    for LayouterCircuit<F, C, Fn>
{
    type ProofSystem = DefaultProofSystem;

    fn generate_circuit_description(
        &mut self,
    ) -> Result<
        (
            GKRCircuitDescription<F, Self::ProofSystem>,
            InputNodeMap,
            InputLayerHintMap<F>,
            CircuitDescriptionMap,
        ),
        GKRError,
    > {
        let ctx = Context::new();
        let component = (self.witness_builder)(&ctx);
        let nodes = component.yield_nodes();
        let (input_nodes, verifier_challenge_nodes, intermediate_nodes, lookup_nodes, output_nodes) =
            layout(ctx, nodes).unwrap();

        let mut input_layer_id = LayerId::Input(0);
        let mut intermediate_layer_id = LayerId::Layer(0);
        let mut verifier_challenge_layer_id = LayerId::VerifierChallengeLayer(0);

        let mut intermediate_layers = Vec::<CircuitLayerEnum<F>>::new();
        let mut output_layers = Vec::<CircuitMleOutputLayer<F>>::new();
        let mut circuit_description_map = CircuitDescriptionMap::new();
        let mut input_node_to_layer_map = InputNodeMap::new();
        let mut input_layer_hint_map = InputLayerHintMap::<F>::new();

        let mut input_layers = input_nodes
            .iter()
            .map(|input_node| {
                dbg!(&input_layer_id);
                let input_circuit_description = input_node
                    .generate_input_circuit_description(
                        &mut input_layer_id,
                        &mut circuit_description_map,
                    )
                    .unwrap();
                input_node_to_layer_map.add_node(&input_layer_id, &input_node.id());
                input_circuit_description
            })
            .collect_vec();

        verifier_challenge_nodes
            .iter()
            .map(|verifier_challenge_node| {
                let verifier_challenge_layer = verifier_challenge_node
                    .generate_circuit_description::<F>(
                        &mut verifier_challenge_layer_id,
                        &mut circuit_description_map,
                    );
                input_layers.push(CircuitInputLayerEnum::RandomInputLayer(
                    verifier_challenge_layer,
                ))
            })
            .collect_vec();

        for node in &intermediate_nodes {
            dbg!(&intermediate_layer_id);
            let node_compiled_intermediate_layers = node
                .generate_circuit_description(
                    &mut intermediate_layer_id,
                    &mut circuit_description_map,
                )
                .unwrap();
            intermediate_layers.extend(node_compiled_intermediate_layers);
        }

        (input_layers, intermediate_layers, output_layers) = lookup_nodes.iter().fold(
            (input_layers, intermediate_layers, output_layers),
            |(mut lookup_input_acc, mut lookup_intermediate_acc, mut lookup_output_acc),
             lookup_node| {
                let (input_layers, intermediate_layers, output_layers) = lookup_node
                    .generate_lookup_circuit_description(
                        &mut input_layer_id,
                        &mut intermediate_layer_id,
                        &mut circuit_description_map,
                        &mut input_layer_hint_map,
                    )
                    .unwrap();
                lookup_input_acc.extend(input_layers);
                lookup_intermediate_acc.extend(intermediate_layers);
                lookup_output_acc.extend(output_layers);
                (lookup_input_acc, lookup_intermediate_acc, lookup_output_acc)
            },
        );

        output_layers =
            output_nodes
                .iter()
                .fold(output_layers, |mut output_layer_acc, output_node| {
                    output_layer_acc
                        .extend(output_node.compile_output(&mut circuit_description_map));
                    output_layer_acc
                });

        let circuit_description =
            GKRCircuitDescription::new(input_layers, intermediate_layers, output_layers);

        Ok((
            circuit_description,
            input_node_to_layer_map,
            input_layer_hint_map,
            circuit_description_map,
        ))
    }

    fn populate_circuit(
        &mut self,
        gkr_circuit_description: GKRCircuitDescription<F, Self::ProofSystem>,
        input_node_to_layer_map: InputNodeMap,
        input_layer_hint_map: InputLayerHintMap<F>,
        data_input_layers: Vec<InputLayerData<F>>,
        circuit_description_map: &CircuitDescriptionMap,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        let GKRCircuitDescription {
            input_layers,
            intermediate_layers,
            output_layers,
        } = gkr_circuit_description;

        // Forward pass through input layer data to map input layer ID to the data that the circuit builder provides.
        let mut input_id_data_map = HashMap::<NodeId, &InputLayerData<F>>::new();
        data_input_layers.iter().for_each(|input_layer_data| {
            input_id_data_map.insert(
                input_layer_data.corresponding_input_node_id,
                input_layer_data,
            );
        });

        // Forward pass to get the map of circuit MLEs whose data is expected to be "compiled"
        // for future layers.
        let mut mle_claim_map = HashMap::<LayerId, Vec<&CircuitMle<F>>>::new();
        intermediate_layers.iter().for_each(|intermediate_layer| {
            let layer_source_circuit_mles = intermediate_layer.get_circuit_mles();
            layer_source_circuit_mles
                .into_iter()
                .for_each(|circuit_mle| {
                    let layer_id = circuit_mle.layer_id();
                    if mle_claim_map.get(&layer_id).is_none() {
                        mle_claim_map.insert(layer_id, vec![circuit_mle]);
                    } else {
                        mle_claim_map.get_mut(&layer_id).unwrap().push(&circuit_mle);
                    }
                })
        });

        output_layers.iter().for_each(|output_layer| {
            let layer_id = output_layer.layer_id();
            if mle_claim_map.get(&layer_id).is_none() {
                mle_claim_map.insert(layer_id, vec![&output_layer.mle]);
            } else {
                mle_claim_map
                    .get_mut(&layer_id)
                    .unwrap()
                    .push(&output_layer.mle);
            }
        });

        let mut circuit_map = CircuitMap::new();

        // input layers
        // go through input data, map it to the inputlayernode it corresponds to
        // for each input layer node, take the input data it corresponds to and combine it to form one big bookkeeping table,
        // we convert the circuit input layer into a prover input layer using this big bookkeeping table
        // we add the data in the input data corresopnding with the circuit location for each input data struct into the circuit map
        let mut prover_input_layers: Vec<InputLayerEnum<F>> = Vec::new();
        input_layers.iter().for_each(|input_layer_description| {
            let input_layer_id = input_layer_description.layer_id();
            let input_node_id = input_node_to_layer_map.get_layer_id(&input_layer_id);
            if input_id_data_map.contains_key(input_node_id) {
                let corresponding_input_data = *(input_id_data_map.get(input_node_id).unwrap());
                let input_mles = corresponding_input_data
                    .data
                    .iter()
                    .map(|input_shred_data| {
                        let (input_shred_circuit_location, input_shred_num_vars) =
                            circuit_description_map
                                .get_node(&input_shred_data.corresponding_input_shred_id)
                                .unwrap();
                        assert_eq!(input_shred_num_vars, &input_shred_data.data.num_vars());
                        circuit_map.add_node(
                            input_shred_circuit_location.clone(),
                            input_shred_data.data.clone(),
                        );
                        &input_shred_data.data
                    });

                let combined_mle = combine_input_mles(&input_mles.collect_vec());
                let mut prover_input_layer = input_layer_description
                    .into_prover_input_layer(combined_mle, &corresponding_input_data.precommit);
                let commitment = prover_input_layer.commit().unwrap();
                InputLayerEnum::append_commitment_to_transcript(&commitment, transcript_writer);
                prover_input_layers.push(prover_input_layer);
            } else {
                if let CircuitInputLayerEnum::RandomInputLayer(
                    verifier_challenge_input_layer_description,
                ) = input_layer_description
                {
                    let verifier_challenge_mle =
                        MultilinearExtension::new(transcript_writer.get_challenges(
                            "Verifier challenges for fiat shamir",
                            verifier_challenge_input_layer_description.num_bits,
                        ));
                    circuit_map.add_node(
                        CircuitLocation::new(
                            verifier_challenge_input_layer_description.layer_id(),
                            vec![],
                        ),
                        verifier_challenge_mle.clone(),
                    );
                    let verifier_challenge_layer = input_layer_description
                        .into_prover_input_layer(verifier_challenge_mle, &None);
                    prover_input_layers.push(verifier_challenge_layer);
                } else {
                    assert!(input_layer_hint_map.0.contains_key(&input_layer_id));
                }
            }
        });

        // forward pass of the layers
        // convert the circuit layer into a prover layer using circuit map -> populate a GKRCircuit as you do this
        // prover layer ( mle_claim_map ) -> populates circuit map
        let mut prover_intermediate_layers: Vec<LayerEnum<F>> = Vec::new();
        intermediate_layers
            .iter()
            .for_each(|intermediate_layer_description| {
                let prover_intermediate_layer =
                    intermediate_layer_description.into_prover_layer(&circuit_map);
                let mle_outputs_necessary = mle_claim_map
                    .get(&intermediate_layer_description.layer_id())
                    .unwrap();
                prover_intermediate_layer
                    .compute_data_outputs(mle_outputs_necessary, &mut circuit_map);
                prover_intermediate_layers.push(prover_intermediate_layer);
            });

        todo!()
    }
}

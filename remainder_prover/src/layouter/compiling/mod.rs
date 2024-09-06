//! A module for defining how certain nodes can be Compiled into a GKR Witness

#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use crate::input_layer::enum_input_layer::CircuitInputLayerEnum;
use crate::layer::layer_enum::CircuitLayerEnum;
use crate::output_layer::mle_output_layer::CircuitMleOutputLayer;
use crate::prover::{GKRCircuitDescription, GKRError};
use crate::{
    layer::LayerId,
    layouter::layouting::layout,
    prover::{proof_system::DefaultProofSystem, GKRCircuit},
};
use itertools::Itertools;
use remainder_shared_types::FieldExt;

use super::layouting::CircuitDescriptionMap;
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
    ) -> Result<GKRCircuitDescription<F, Self::ProofSystem>, GKRError> {
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

        let mut input_layers = input_nodes
            .iter()
            .map(|input_node| {
                dbg!(&input_layer_id);
                input_node
                    .generate_input_circuit_description(
                        &mut input_layer_id,
                        &mut circuit_description_map,
                    )
                    .unwrap()
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

        Ok(circuit_description)
    }
}

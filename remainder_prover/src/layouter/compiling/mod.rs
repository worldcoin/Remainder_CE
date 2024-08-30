//! A module for defining how certain nodes can be Compiled into a GKR Witness

#[cfg(test)]
mod tests;

use std::iter;
use std::marker::PhantomData;

use crate::input_layer::enum_input_layer::InputLayerEnum;
use crate::input_layer::InputLayer;
use crate::layer::layer_enum::LayerEnum;
use crate::layer::Layer;
use crate::output_layer::mle_output_layer::MleOutputLayer;
use crate::prover::{GKRError, WitnessAndCircuitDescription};
use crate::{
    layer::LayerId,
    layouter::layouting::layout,
    prover::{
        layers::Layers,
        proof_system::{DefaultProofSystem, ProofSystem},
        GKRCircuit, Witness,
    },
};
use itertools::Itertools;
use remainder_shared_types::transcript::ProverTranscript;
use remainder_shared_types::FieldExt;

use super::nodes::circuit_inputs::InputLayerNode;
use super::nodes::random::VerifierChallengeNode;
use super::nodes::CompilableNode;
use super::{
    component::Component,
    layouting::CircuitMap,
    nodes::{node_enum::NodeEnum, Context},
};

/// An intermediate struct that allows a `Witness` to be built
/// one layer at a time
#[derive(Clone, Debug, Default)]
pub struct WitnessBuilder<F: FieldExt, Pf: ProofSystem<F>> {
    input_layers: Vec<Pf::InputLayer>,
    layers: Layers<F, Pf::Layer>,
    output_layers: Vec<Pf::OutputLayer>,
}

impl<F: FieldExt, Pf: ProofSystem<F>> WitnessBuilder<F, Pf> {
    /// Creates an empty `WitnessBuilder`
    pub fn new() -> Self {
        Self {
            input_layers: Vec::new(),
            layers: Layers::new(),
            output_layers: Vec::new(),
        }
    }

    /// Gives the expected `LayerId` of the next InputLayer
    pub fn next_input_layer(&self) -> LayerId {
        self.input_layers
            .last()
            .map(|last| last.layer_id().next())
            .unwrap_or(LayerId::Input(0))
    }

    /// Gives the expected `LayerId` of the next Layer
    pub fn next_layer(&self) -> LayerId {
        self.layers
            .layers
            .last()
            .map(|last| last.layer_id().next())
            .unwrap_or(LayerId::Layer(0))
    }

    /// Adds an InputLayer
    pub fn add_input_layer(&mut self, input_layer: <Pf as ProofSystem<F>>::InputLayer) {
        self.input_layers.push(input_layer);
    }

    /// Adds a Layer
    pub fn add_layer(&mut self, layer: <Pf as ProofSystem<F>>::Layer) {
        self.layers.layers.push(layer);
    }

    /// Adds an OutputLayer
    pub fn add_output_layer(&mut self, output_layer: <Pf as ProofSystem<F>>::OutputLayer) {
        self.output_layers.push(output_layer);
    }

    /// Builds a Witness that is ready to be proven
    pub fn build(self) -> Witness<F, Pf> {
        Witness {
            layers: self.layers,
            output_layers: self.output_layers,
            input_layers: self.input_layers,
        }
    }
}

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
    // write random
    // add the input node depednency into

    // {
    //     let mut witness = self.synthesize();

    //     let verifier_key = witness.generate_verifier_key()?;

    //     let commitments = witness
    //         .input_layers
    //         .iter_mut()
    //         .map(|input_layer| {
    //             let commitment = input_layer.commit().map_err(GKRError::InputLayerError)?;
    //             CircuitInputLayer::<F, Self>::append_commitment_to_transcript(
    //                 &commitment,
    //                 transcript,
    //             );
    //             Ok(commitment)
    //         })
    //         .try_collect()?;

    //     Ok((witness, commitments, verifier_key))
    // }

    fn synthesize_and_commit(
        &mut self,
        transcript: &mut impl ProverTranscript<F>,
    ) -> Result<WitnessAndCircuitDescription<F, Self>, GKRError> {
        let ctx = Context::new();
        let component = (self.witness_builder)(&ctx);
        let nodes = component.yield_nodes();
        let (
            input_nodes,
            mut verifier_challenge_nodes,
            intermediate_nodes,
            lookup_nodes,
            output_nodes,
        ) = layout(ctx, nodes).unwrap();

        let mut input_layer_id = LayerId::Input(0);
        let mut intermediate_layer_id = LayerId::Layer(0);
        let mut verifier_challenge_layer_id = LayerId::VerifierChallengeLayer(0);

        let mut intermediate_layers = Vec::<LayerEnum<F>>::new();
        let mut output_layers = Vec::<MleOutputLayer<F>>::new();
        let mut circuit_map = CircuitMap::new();

        let mut input_layers = input_nodes
            .iter()
            .map(|input_node| {
                dbg!(&input_layer_id);
                input_node
                    .compile_input(&mut input_layer_id, &mut circuit_map, transcript)
                    .unwrap()
            })
            .collect_vec();

        let verifier_challenge_layers = verifier_challenge_nodes
            .iter_mut()
            .map(|verifier_challenge_node| {
                dbg!(&input_layer_id);
                verifier_challenge_node.compile(&mut input_layer_id, &mut circuit_map, transcript)
            })
            .collect_vec();

        for node in &intermediate_nodes {
            dbg!(&intermediate_layer_id);
            let node_compiled_intermediate_layers = node
                .compile(&mut intermediate_layer_id, &mut circuit_map)
                .unwrap();
            intermediate_layers.extend(node_compiled_intermediate_layers);
        }

        (input_layers, intermediate_layers, output_layers) = lookup_nodes.iter().fold(
            (input_layers, intermediate_layers, output_layers),
            |(mut lookup_input_acc, mut lookup_intermediate_acc, mut lookup_output_acc),
             lookup_node| {
                let (input_layers, intermediate_layers, output_layers) = lookup_node
                    .compile_lookup(
                        &mut input_layer_id,
                        &mut intermediate_layer_id,
                        &mut circuit_map,
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
                    output_layer_acc.extend(output_node.compile_output(&mut circuit_map));
                    output_layer_acc
                });

        let instantiated_circuit: Witness<F, DefaultProofSystem> = Witness {
            input_layers: input_layers,
            layers: Layers::new_with_layers(intermediate_layers),
            output_layers: output_layers,
        };

        let circuit_description = instantiated_circuit.generate_verifier_key().unwrap();

        Ok((instantiated_circuit, circuit_description))
    }
}

//! A module for defining how certain nodes can be Compiled into a GKR Witness

#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    input_layer::InputLayer,
    layer::{Layer, LayerId},
    layouter::layouting::layout,
    prover::{
        layers::Layers,
        proof_system::{DefaultProofSystem, ProofSystem},
        GKRCircuit, Witness,
    },
};

use super::{
    component::Component,
    layouting::CircuitMap,
    nodes::{node_enum::NodeEnum, Context},
};

/// An intermediate struct that allows a `Witness` to be built
/// one layer at a time
#[derive(Clone, Debug)]
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
            .map(|last| last.id().next())
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

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let ctx = Context::new();
        let component = (&mut self.witness_builder)(&ctx);
        let nodes = component.yield_nodes();
        let compiled_nodes = layout(ctx, nodes).unwrap();

        let mut witness_builder = WitnessBuilder::new();
        let mut circuit_map = CircuitMap::new();

        for node in &compiled_nodes {
            node.compile(&mut witness_builder, &mut circuit_map)
                .unwrap()
        }
        witness_builder.build()
    }
}

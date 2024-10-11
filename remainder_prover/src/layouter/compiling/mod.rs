//! This is a module for constructing an [InstantiatedCircuit] given a
//! [LayouterCircuit], which is a closure provided by the circuit builder that
//! takes in one parameter, the context, along with all of the "circuit
//! builders" necessary in order to represent the data-dependency relationships
//! within the circuit. Additionally, the circuit builder in this closure also
//! provides the necessary input data to populate the circuits with.
//!
//! The important distinction in this compilation process is that first, using
//! the data-dependency relationships in the circuit, a [GKRCircuitDescription]
//! is generated, which represents the data-dependencies in the circuit along
//! with the shape of the circuit itself (the number of variables in the
//! different MLEs that make up a layer).
//!
//! Then, using this circuit description along with the data inputs, the
//! `instantiate()` function will create an [InstantiatedCircuit], where now
//! the circuit description is "filled in" with its associated data.

#[cfg(test)]
mod tests;

use remainder_shared_types::transcript::poseidon_transcript::PoseidonSponge;
use remainder_shared_types::Field;
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::layer::LayerId;
use crate::mle::evals::MultilinearExtension;
use crate::prover::{generate_circuit_description, prove, GKRCircuitDescription, GKRError};
use remainder_shared_types::transcript::{Transcript, TranscriptWriter};

use super::nodes::circuit_inputs::InputLayerNodeData;
use super::nodes::NodeId;
use super::{
    component::Component,
    nodes::{node_enum::NodeEnum, Context},
};

/// Defines the type of hash used when adding a circuit description's hash
/// into the transcript.
pub enum CircuitHashType {
    /// This uses Rust's [DefaultHasher] implementation and uses the
    /// #[derive(Hash)] implementation. The hash function implemented
    /// underneath is not cryptographically secure, and thus this option
    /// is generally not recommended.
    DefaultRustHash,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes using [Sha3_256].
    Sha3_256,
    /// This converts the circuit description into a JSON string via
    /// [serde::Serialize] and hashes the bytes in chunks of 16, converting
    /// them first to field elements.
    Poseidon,
}

/// The struct used in order to aid with proving, which contains the function
/// used in order to generate the circuit description itself, called the
/// `witness_builder`, which can then be used to generate an
/// [InstantiatedCircuit].
pub struct LayouterCircuit<
    F: Field,
    C: Component<NodeEnum<F>>,
    Fn: FnMut(&Context) -> (C, Vec<InputLayerNodeData<F>>),
> {
    witness_builder: Fn,
    _marker: PhantomData<F>,
}

impl<
        F: Field,
        C: Component<NodeEnum<F>>,
        Fn: FnMut(&Context) -> (C, Vec<InputLayerNodeData<F>>),
    > LayouterCircuit<F, C, Fn>
{
    /// Constructs a `LayouterCircuit` by taking in a closure whose parameter is
    /// a [Context], which determines the ID of each individual part of the
    /// circuit whose data output will be computed for future layers. The
    /// function itself returns a [Component], which is a set of circuit "nodes"
    /// which will be used to build the circuit itself. The closure also returns
    /// a [Vec<InputLayerNodeData>], which is a vector of all of the data that will
    /// be fed into the input layers of the circuit.
    pub fn new(witness_builder: Fn) -> Self {
        Self {
            witness_builder,
            _marker: PhantomData,
        }
    }

    /// From the witness builder, generate a circuit description and then an
    /// instantiated circuit. Then, write the "proof" of this circuit to the
    /// `transcript_writer`. Return the circuit description for verifying.
    pub fn prove(
        &mut self,
        mut transcript_writer: TranscriptWriter<F, PoseidonSponge<F>>,
    ) -> Result<
        (
            Transcript<F>,
            GKRCircuitDescription<F>,
            HashMap<LayerId, MultilinearExtension<F>>,
        ),
        GKRError,
    > {
        let ctx = Context::new();
        let (component, input_layer_data) = (self.witness_builder)(&ctx);

        // Convert the input layer data into a map that maps the input shred ID
        // i.e. adapt witness builder output to the instantate() function.
        // This can be removed once witness builders are removed.
        let mut shred_id_to_data = HashMap::<NodeId, MultilinearExtension<F>>::new();
        input_layer_data.into_iter().for_each(|input_layer_data| {
            input_layer_data
                .data
                .into_iter()
                .for_each(|input_shred_data| {
                    shred_id_to_data.insert(
                        input_shred_data.corresponding_input_shred_id,
                        input_shred_data.data,
                    );
                });
        });

        let (circuit_description, input_builder, _) =
            generate_circuit_description(component.yield_nodes()).unwrap();

        let inputs = input_builder(shred_id_to_data).unwrap();
        prove(
            &inputs,
            &HashMap::new(),
            &circuit_description,
            &mut transcript_writer,
        )
        .unwrap();

        Ok((
            transcript_writer.get_transcript(),
            circuit_description,
            inputs,
        ))
    }
}

pub mod sumcheck_layer;

use serde::{Deserialize, Serialize};
use std::fmt::{Debug};

use crate::{
    claims::Claim,
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};

/// A layer is the smallest component of the GKR protocol.
///
/// Each `Layer` is a sub-protocol that takes in some `Claim` and creates a proof
/// that the `Claim` is correct
pub trait Layer<F: FieldExt> {
    /// The struct that contains the proof this `Layer` generates
    type Proof: Debug + Serialize + for<'a> Deserialize<'a>;

    type Error: std::error::Error;

    /// Creates a proof for this Layer
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript: &mut impl ProverTranscript<F>,
    ) -> Result<Self::Proof, Self::Error>;

    /// Verifies the `Layer`'s proof
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        proof: Self::Proof,
        transcript: &mut impl VerifierTranscript<F>,
    ) -> Result<(), Self::Error>;

    /// Gets this `Layer`'s `LayerId`
    fn id(&self) -> &LayerId;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy, PartialOrd)]
/// The location of a layer within the GKR circuit
pub enum LayerId {
    /// A random mle input layer
    ///
    /// TODO!(nick) Remove this once new batching code is implemented
    RandomInput(usize),
    /// An Mle located in the input layer
    Input(usize),
    /// A layer between the output layer and input layers
    Layer(usize),
    /// An MLE located in the output layer.
    Output(usize),
}

impl LayerId {
    /// Gets a new LayerId which represents a layerid of the same type but with an incremented id number
    pub fn next(&self) -> LayerId {
        match self {
            LayerId::RandomInput(id) => LayerId::RandomInput(id + 1),
            LayerId::Input(id) => LayerId::Input(id + 1),
            LayerId::Layer(id) => LayerId::Layer(id + 1),
            LayerId::Output(id) => LayerId::Output(id + 1),
        }
    }
}

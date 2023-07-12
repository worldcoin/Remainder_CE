//!Module that orchestrates creating a GKR Proof

pub trait GKRProver<F: FieldExt> {
    type Transcript;

    fn new() -> Self;

    fn add_layer(&mut self, layer: &impl Layer<F>);

    fn prove(&mut self, transcript: &mut Self::Transcript);
}


use std::{marker::PhantomData, path::Path};

use crate::{FieldExt, prover::GKRCircuit, transcript::poseidon_transcript::PoseidonTranscript};

///GKRCircuit that proves inference for a single decision tree
pub struct ZKDTCircuit<F: FieldExt> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> ZKDTCircuit<F> {
    pub fn new(directory: &Path) -> Self {
        todo!()
    }
}

impl<F: FieldExt> GKRCircuit<F> for ZKDTCircuit<F> {
    type Transcript = PoseidonTranscript<F>;

    fn synthesize(&mut self) -> (crate::prover::Layers<F, Self::Transcript>, Vec<crate::prover::OutputLayer<F>>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {

}
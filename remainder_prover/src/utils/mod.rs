/// Helpful functions for manipulating ndarray Array objects (e.g. padding)
pub mod array;
/// Helpful arithmetic functions.
pub mod arithmetic;
/// Helpful functions for manipulating MLEs (e.g. padding).
pub mod mle;
/// Helpful functions for debugging.
pub mod debug;

#[cfg(test)]
/// Utilities that are only useful for tests
pub(crate) mod test_utils;

/// FIXME the functions below are uncategorized and probably should be moved to a more appropriate
/// module or submodule.

use remainder_shared_types::{FieldExt, Poseidon};

use crate::{
    layer::layer_enum::LayerEnum,
    prover::layers::Layers,
};


/// Hashes the layers of a GKR circuit by calling their circuit descriptions
/// Returns one single Field element
pub fn hash_layers<F: FieldExt>(layers: &Layers<F, LayerEnum<F>>) -> F {
    let mut sponge: Poseidon<F, 3, 2> = Poseidon::new(8, 57);

    layers.layers.iter().for_each(|layer| {
        let item = format!("{}", layer.circuit_description_fmt());
        let bytes = item.as_bytes();
        let elements: Vec<F> = bytes
            .chunks(62)
            .map(|bytes| {
                let base = F::from(8);
                let first = bytes[0];
                bytes
                    .iter()
                    .skip(1)
                    .fold((F::from(first as u64), base), |(accum, power), byte| {
                        let accum = accum + (F::from(*byte as u64) * power);
                        let power = power * base;
                        (accum, power)
                    })
                    .0
            })
            .collect::<Vec<_>>();

        sponge.update(&elements);
    });

    sponge.squeeze()
}

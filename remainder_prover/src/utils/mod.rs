//! Module for useful functions
/// Helpful arithmetic functions.
pub mod arithmetic;
/// Helpful functions for manipulating ndarray Array objects (e.g. padding)
pub mod array;
/// Helpful functions for debugging.
pub mod debug;
/// Helpful functions for manipulating MLEs (e.g. padding).
pub mod mle;

#[cfg(test)]
/// Utilities that are only useful for tests
pub(crate) mod test_utils;

use std::fs;

/// FIXME the functions below are uncategorized and probably should be moved to a more appropriate
/// module or submodule.
use remainder_shared_types::{Field, Poseidon};

use crate::{
    layer::layer_enum::LayerEnum,
    layouter::nodes::{
        circuit_inputs::{InputLayerNode, InputShred, InputShredData},
        CircuitNode, Context,
    },
    mle::evals::MultilinearExtension,
    prover::layers::Layers,
};

/// Using the number of variables, get an input shred that represents
/// this information.
pub fn get_input_shred_from_num_vars(
    num_vars: usize,
    ctx: &Context,
    input_node: &InputLayerNode,
) -> InputShred {
    InputShred::new(ctx, num_vars, input_node)
}

/// Using a data vector, get an [InputShred] which represents its
/// shape, along with [InputShredData] which represents the
/// corresponding data.
pub fn build_input_shred_and_data<F: Field>(
    data: MultilinearExtension<F>,
    ctx: &Context,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<F>) {
    let input_shred = InputShred::new(ctx, data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Using a data vector, get an [InputShred] which represents its
/// shape, along with [InputShredData] which represents the
/// corresponding data.
pub fn get_input_shred_and_data<F: Field>(
    mle_vec: Vec<F>,
    ctx: &Context,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<F>) {
    assert!(mle_vec.len().is_power_of_two());
    let data = MultilinearExtension::new(mle_vec);
    let input_shred = InputShred::new(ctx, data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Returns whether a particular file exists in the filesystem
///
/// TODO!(ryancao): Shucks does this check a relative path...?
pub fn file_exists(file_path: &String) -> bool {
    match fs::metadata(file_path) {
        Ok(file_metadata) => file_metadata.is_file(),
        Err(_) => false,
    }
}

/// Hashes the layers of a GKR circuit by calling their circuit descriptions
/// Returns one single Field element
pub fn hash_layers<F: Field>(layers: &Layers<F, LayerEnum<F>>) -> F {
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

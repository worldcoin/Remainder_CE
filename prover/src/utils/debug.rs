use std::collections::HashMap;

use num::Integer;
use shared_types::Field;

use crate::{claims::Claim, layer::LayerId, prover::GKRCircuitDescription};

/// Prints all claims on input layers, aggregated by input layer, along with
/// the number of variables in each input layer where a claim is made.
///
/// Purely for sanitycheck against input layers and ensuring that # claims
/// roughly matches expected # claims.
pub fn sanitycheck_input_layers_and_claims<F: Field>(
    input_layer_claims: &[Claim<F>],
    circuit_description: &GKRCircuitDescription<F>,
) {
    let mut input_layer_claims_map: HashMap<LayerId, usize> = HashMap::new();
    input_layer_claims.iter().for_each(|claim| {
        input_layer_claims_map
            .entry(claim.get_to_layer_id())
            .and_modify(|x| x.inc())
            .or_insert(1);
    });
    // Mapping between input layer IDs and the number of variables in that input
    // layer's representation.
    let input_layers_map: HashMap<LayerId, usize> = circuit_description
        .input_layers
        .iter()
        .map(|circuit_input_layer| (circuit_input_layer.layer_id, circuit_input_layer.num_vars))
        .collect();
    input_layer_claims_map
        .iter()
        .for_each(|(layer_id, num_claims)| {
            let layer_num_vars = input_layers_map.get(layer_id).unwrap();
            println!("Layer ID {layer_id} with {num_claims} claims and {layer_num_vars} num vars");
        });
}

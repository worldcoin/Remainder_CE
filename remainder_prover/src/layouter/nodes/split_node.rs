//! A node that can alter the claims made on it's source `ClaimableNode`

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::{
    layer::layer_enum::LayerDescriptionEnum,
    layouter::layouting::{CircuitDescriptionMap, CircuitLocation, DAGError},
};

use super::{CircuitNode, CompilableNode, NodeId};

/// A Node that derives new `ClaimableNode`s from a single
/// `ClaimableNode`.
///
/// The new nodes represent the input node split
/// by a selector bit.
#[derive(Clone, Debug)]
pub struct SplitNode {
    id: NodeId,
    num_vars: usize,
    source: NodeId,
    prefix_bits: Vec<bool>,
}

impl SplitNode {
    /// Creates 2^num_vars `SplitNodes` from a single ClaimableNode
    pub fn new(node: &impl CircuitNode, num_vars: usize) -> Vec<Self> {
        let num_vars_node = node.get_num_vars();
        let source = node.id();
        let max_num_vars = num_vars_node - num_vars;
        (0..(1 << num_vars))
            .zip(bits_iter(num_vars))
            .map(|(_, prefix_bits)| Self {
                id: NodeId::new(),
                source,
                num_vars: max_num_vars,
                prefix_bits,
            })
            .collect()
    }
}

impl CircuitNode for SplitNode {
    fn id(&self) -> NodeId {
        self.id
    }

    fn sources(&self) -> Vec<NodeId> {
        vec![self.source]
    }

    fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> CompilableNode<F> for SplitNode {
    fn generate_circuit_description(
        &self,
        circuit_description_map: &mut CircuitDescriptionMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>, DAGError> {
        let (source_location, _) =
            circuit_description_map.get_location_num_vars_from_node_id(&self.source)?;

        let prefix_bits = source_location
            .prefix_bits
            .iter()
            .chain(self.prefix_bits.iter())
            .copied()
            .collect();

        let location = CircuitLocation::new(source_location.layer_id, prefix_bits);

        circuit_description_map
            .add_node_id_and_location_num_vars(self.id, (location, self.get_num_vars()));
        Ok(vec![])
    }
}

///returns an iterator that wil give permutations of binary bits of size
/// num_bits
///
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
fn bits_iter(num_bits: usize) -> impl Iterator<Item = Vec<bool>> {
    std::iter::successors(Some(vec![false; num_bits]), move |prev| {
        let mut prev = prev.clone();
        let mut removed_bits = 0;
        for index in (0..num_bits).rev() {
            let curr = prev.remove(index);
            if !curr {
                prev.push(true);
                break;
            } else {
                removed_bits += 1;
            }
        }
        if removed_bits == num_bits {
            None
        } else {
            Some(
                prev.into_iter()
                    .chain(repeat_n(false, removed_bits))
                    .collect_vec(),
            )
        }
    })
}

#[cfg(test)]
mod test {

    use std::collections::HashMap;

    use ark_std::log2;
    use itertools::Itertools;

    use remainder_shared_types::{Field, Fr};

    use crate::{
        expression::{abstract_expr::AbstractExpr, generic_expr::Expression},
        layer::LayerId,
        layouter::nodes::{
            circuit_inputs::{InputLayerNode, InputShred},
            circuit_outputs::OutputNode,
            node_enum::NodeEnum,
            sector::Sector,
            CircuitNode, NodeId,
        },
        mle::evals::MultilinearExtension,
        prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
    };

    use super::SplitNode;

    /// Struct which allows for easy "semantic" feeding of inputs into the
    /// test split node circuit.
    struct SplitNodeTestInputs<F: Field> {
        mle: MultilinearExtension<F>,
        expected_mle: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving for the split node test circuit.
    fn build_identity_gate_test_circuit_description<F: Field>(
        mle_to_be_split_num_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(SplitNodeTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Sanitycheck ---
        assert!(mle_to_be_split_num_vars > 0);

        // --- All inputs are public inputs ---
        let public_input_layer_node = InputLayerNode::new(None);

        // --- Two inputs to the circuit: the MLE to be split and multiplied against itself,
        // and the expected result of that ---
        let mle_shred = InputShred::new(mle_to_be_split_num_vars, &public_input_layer_node);
        let expected_mle_shred =
            InputShred::new(mle_to_be_split_num_vars - 1, &public_input_layer_node);

        // --- Save IDs to be used later ---
        let mle_shred_id = mle_shred.id();
        let expected_mle_shred_id = expected_mle_shred.id();

        // --- Create the circuit components (this just splits the MLE in half) ---
        let split_sectors = SplitNode::new(&mle_shred, 1);
        let sector_prod = Sector::new(&[&split_sectors[0], &split_sectors[1]], |inputs| {
            Expression::<F, AbstractExpr>::products(vec![inputs[0], inputs[1]])
        });

        let final_sector = Sector::new(&[&&sector_prod, &expected_mle_shred], |inputs| {
            Expression::<F, AbstractExpr>::mle(inputs[0])
                - Expression::<F, AbstractExpr>::mle(inputs[1])
        });

        let output = OutputNode::new_zero(&final_sector);

        // --- Generate the circuit description ---
        let all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            mle_shred.into(),
            expected_mle_shred.into(),
            sector_prod.into(),
            final_sector.into(),
            output.into(),
        ]
        .into_iter()
        .chain(
            split_sectors
                .into_iter()
                .map(|split_node| split_node.into())
                .collect_vec(),
        )
        .collect_vec();

        let (circuit_description, convert_input_shreds_to_input_layers) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |split_node_test_inputs: SplitNodeTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (mle_shred_id, split_node_test_inputs.mle),
                (expected_mle_shred_id, split_node_test_inputs.expected_mle),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
    }

    #[test]
    fn test_split_node_in_circuit() {
        // --- Define all the input (data) to the circuit ---
        let mle = MultilinearExtension::<Fr>::interlace_mles(vec![
            MultilinearExtension::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
            MultilinearExtension::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]),
        ]);
        let mle_num_vars = log2(mle.f.len());
        // Then we should have mle_out = [1*5, 2*6, 3*7, 4*8], the product between the two split nodes
        let mle_out = MultilinearExtension::new(vec![
            Fr::from(5),
            Fr::from(2 * 6),
            Fr::from(3 * 7),
            Fr::from(4 * 8),
        ]);

        // --- Create circuit description + input helper function ---
        let (identity_gate_test_circuit_desc, input_helper_fn) =
            build_identity_gate_test_circuit_description(mle_num_vars as usize);

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(SplitNodeTestInputs {
            mle,
            expected_mle: mle_out,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(
            &identity_gate_test_circuit_desc,
            private_input_layers,
            &circuit_inputs,
        );
    }
}

//! A node that splits a single MLE into 2^num_vars smaller MLEs using prefix bits.

use itertools::{repeat_n, Itertools};
use remainder_shared_types::Field;

use crate::layouter::builder::CircuitMap;
use remainder::{circuit_layout::CircuitLocation, layer::layer_enum::LayerDescriptionEnum};

use super::{CircuitNode, CompilableNode, NodeId};

use anyhow::Result;
/// A node that splits a single MLE into 2^num_vars smaller MLEs using prefix bits.
/// Works big endian.
#[derive(Clone, Debug)]
pub struct SplitNode {
    id: NodeId,
    num_vars: usize,
    source: NodeId,
    prefix_bits: Vec<bool>,
}

impl SplitNode {
    /// Creates 2^num_vars instances of [SplitNode] from a single [CircuitNode] using prefix bits in
    /// big-endian order. For example, if num_vars is 2, the prefix bits of the returned
    /// instances will be (in order): 00, 01, 10, 11.
    pub fn new(node: &dyn CircuitNode, num_vars: usize) -> Vec<Self> {
        let num_vars_node = node.get_num_vars();
        let source = node.id();
        let max_num_vars = num_vars_node - num_vars;
        bits_iter(num_vars)
            .map(|prefix_bits| {
                let prefix_bits = prefix_bits.into_iter().collect();
                Self {
                    id: NodeId::new(),
                    source,
                    num_vars: max_num_vars,
                    prefix_bits,
                }
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
        circuit_map: &mut CircuitMap,
    ) -> Result<Vec<LayerDescriptionEnum<F>>> {
        let (source_location, _) = circuit_map.get_location_num_vars_from_node_id(&self.source)?;

        let prefix_bits = source_location
            .prefix_bits
            .iter()
            .chain(self.prefix_bits.iter())
            .copied()
            .collect();

        let location = CircuitLocation::new(source_location.layer_id, prefix_bits);

        circuit_map.add_node_id_and_location_num_vars(self.id, (location, self.get_num_vars()));
        Ok(vec![])
    }
}

/// Returns an iterator that gives the MSB-first binary representation of the numbers from 0 to
/// 2^num_bits.
/// 0,0,0 -> 0,0,1 -> 0,1,0 -> 0,1,1 -> 1,0,0 -> 1,0,1 -> 1,1,0 -> 1,1,1
/// # Example:
/// ```
/// use remainder_frontend::layouter::nodes::split_node::bits_iter;
/// let bits_iter = bits_iter(2);
/// let bits: Vec<Vec<bool>> = bits_iter.collect();
/// assert_eq!(bits, vec![
///   vec![false, false],
///   vec![false, true],
///   vec![true, false],
///   vec![true, true],
/// ]);
/// ```
pub fn bits_iter(num_bits: usize) -> impl Iterator<Item = Vec<bool>> {
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
    use crate::{
        abstract_expr::AbstractExpression,
        layouter::builder::{Circuit, CircuitBuilder, LayerVisibility},
    };
    use remainder::{
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
    };
    use remainder_shared_types::{Field, Fr};

    // Build a circuit that takes in a single input with 8 values, splits it into four, and
    // subtracts the first two from one another (this is the output).
    fn build_basic_split_circuit<F: Field>() -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        let input_layer = builder.add_input_layer(LayerVisibility::Public);
        let input = builder.add_input_shred("Input", 3, &input_layer);
        let splits = builder.add_split_node(&input, 2);
        let subtractor = builder.add_sector(&splits[0] - &splits[1]);
        builder.set_output(&subtractor);

        builder.build().unwrap()
    }

    #[test]
    #[should_panic]
    fn test_that_split_node_works_little_endian() {
        let a = 1;
        let b = 2;
        // the following values work if SplitNode is LITTLE endian
        let values: Vec<u64> = vec![a, a, 111, 1111, b, b, 11111, 111111];
        let mle = MultilinearExtension::new(values.into_iter().map(|v| Fr::from(v)).collect());
        let mut circuit = build_basic_split_circuit::<Fr>();
        circuit.set_input("Input", mle);
        let provable_circuit = circuit.finalize().unwrap();
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    #[test]
    fn test_that_split_node_works_big_endian() {
        // the following values work if SplitNode is BIG endian
        let values: Vec<u64> = vec![11, 2, 11, 2, 123, 124, 125, 126];
        let mle = MultilinearExtension::new(values.into_iter().map(|v| Fr::from(v)).collect());
        let mut circuit = build_basic_split_circuit::<Fr>();
        circuit.set_input("Input", mle);
        let provable_circuit = circuit.finalize().unwrap();
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    // Build a circuit that takes 4 MLEs, joins them using selectors, splits them into 4 MLEs using
    // SplitNode, and checks that this is the noop.
    fn build_splits_and_selectors_circuit<F: Field>() -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        let num_vars = 2;
        let input_layer = builder.add_input_layer(LayerVisibility::Public);
        let input0 = builder.add_input_shred("Input 0", num_vars, &input_layer);
        let input1 = builder.add_input_shred("Input 1", num_vars, &input_layer);
        let input2 = builder.add_input_shred("Input 2", num_vars, &input_layer);
        let input3 = builder.add_input_shred("Input 3", num_vars, &input_layer);

        let concatenator = builder.add_sector(AbstractExpression::binary_tree_selector(vec![
            &input0, &input1, &input2, &input3,
        ]));

        let splits = builder.add_split_node(&concatenator, num_vars);
        assert_eq!(splits.len(), 4);

        let subtractor = builder.add_sector(
            (&splits[0] - input0)
                + (&splits[1] - input1)
                + (&splits[2] - input2 + (&splits[3] - input3)),
        );
        builder.set_output(&subtractor);

        builder.build().unwrap()
    }

    #[test]
    // Test that SplitNode undoes what the work of selector bits.
    fn test_splits_and_selectors() {
        let mut circuit = build_splits_and_selectors_circuit::<Fr>();

        circuit.set_input(
            "Input 0",
            MultilinearExtension::new(vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]),
        );
        circuit.set_input(
            "Input 1",
            MultilinearExtension::new(vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)]),
        );
        circuit.set_input(
            "Input 2",
            MultilinearExtension::new(vec![Fr::from(9), Fr::from(10), Fr::from(11), Fr::from(12)]),
        );
        circuit.set_input(
            "Input 3",
            MultilinearExtension::new(vec![Fr::from(13), Fr::from(14), Fr::from(15), Fr::from(16)]),
        );

        let provable_circuit = circuit.finalize().unwrap();
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}

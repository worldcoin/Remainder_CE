use itertools::Itertools;
use quickcheck::Arbitrary;
use remainder::{layer::layer_enum::LayerDescriptionEnum, utils::arithmetic::log2_ceil};
use shared_types::{Field, Fr};

use crate::layouter::builder::{Circuit, CircuitBuilder, LayerVisibility};

#[derive(Clone, Debug)]
struct QuickCheckCircuit<F: Field> {
    circuit: Circuit<F>,
    maybe_max_log_layer_size: Option<usize>,
}

const NUM_SECTORS_IN_COMBINED_LAYER: usize = 500;
impl Arbitrary for QuickCheckCircuit<Fr> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Initial specifications.
        let num_sectors = NUM_SECTORS_IN_COMBINED_LAYER;
        let min_vars_in_sector = 18;
        let max_vars_in_sector = 22;
        let mut total_num_coeffs: usize = 0;
        let mut builder = CircuitBuilder::<Fr>::new();

        // 3/4 probability we want the max layer size to be something.
        let should_specify_max_layer_size = *g.choose(&[true, true, true, false]).unwrap();
        (0..num_sectors).for_each(|i| {
            let num_vars_in_sector = *g
                .choose(&(min_vars_in_sector..max_vars_in_sector).collect_vec())
                .unwrap();
            let input_node_ref =
                builder.add_input_layer(&format!("Input Layer {i}"), LayerVisibility::Public);
            let input_shred = builder.add_input_shred(
                &format!("testing_shred {:?}", i),
                num_vars_in_sector,
                &input_node_ref,
            );
            let sector = builder.add_sector(input_shred.expr() + input_shred.expr());
            let output_sector = builder.add_sector(sector.expr() - sector.expr());
            builder.set_output(&output_sector);
            total_num_coeffs += 1 << num_vars_in_sector;
        });
        let maybe_max_log_layer_size = if should_specify_max_layer_size {
            let mut log_layer_size_options = vec![0];
            log_layer_size_options.extend(max_vars_in_sector..log2_ceil(total_num_coeffs) as usize);
            Some(*g.choose(&log_layer_size_options).unwrap() as usize)
        } else {
            None
        };

        Self {
            circuit: builder
                .build_with_max_layer_size(maybe_max_log_layer_size)
                .unwrap(),
            maybe_max_log_layer_size,
        }
    }
}

#[quickcheck]
fn test_combine_with_max_layer_size(quickcheck_circuit: QuickCheckCircuit<Fr>) {
    let circuit = quickcheck_circuit.circuit;
    let description = circuit.get_circuit_description();
    let intermediate_layers = &description.intermediate_layers;
    if quickcheck_circuit.maybe_max_log_layer_size.is_some() {
        let max_log_layer_size = quickcheck_circuit.maybe_max_log_layer_size.unwrap();
        if max_log_layer_size == 0 {
            assert_eq!(intermediate_layers.len(), NUM_SECTORS_IN_COMBINED_LAYER * 2);
        } else {
            for layer in intermediate_layers {
                let num_vars_layer = match layer {
                    LayerDescriptionEnum::Regular(regular_layer) => regular_layer.get_num_vars(),
                    _ => {
                        panic!("Should not have any other layer for this test.")
                    }
                };
                assert!(num_vars_layer <= max_log_layer_size);
            }
        }
    } else {
        assert_eq!(intermediate_layers.len(), 2)
    }
}

use remainder::{
    mle::evals::{Evaluations, MultilinearExtension},
    prover::helpers::test_circuit_with_runtime_optimized_config,
};
use remainder_frontend::layouter::builder::{CircuitBuilder, LayerVisibility};
use remainder_shared_types::Fr;

#[test]
fn test_basic_circuit() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);

    let input_shred_1 = builder.add_input_shred("Input 1", 2, &input_layer);
    let input_shred_1_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![Fr::one(), Fr::one(), Fr::one(), Fr::one()],
    ));
    let input_shred_2 = builder.add_input_shred("Input 2", 2, &input_layer);
    let input_shred_2_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![Fr::from(16), Fr::from(16), Fr::from(16), Fr::from(16)],
    ));

    let sector_1 = builder.add_sector(&input_shred_1 + &input_shred_2);

    let sector_2 = builder.add_sector(&input_shred_1 * &input_shred_2);

    let out_sector = builder.add_sector(sector_1 * sector_2);

    let expected_output_shred = builder.add_input_shred("Expected Output", 2, &input_layer);
    let expected_output_data = MultilinearExtension::new_from_evals(Evaluations::new(
        2,
        vec![
            Fr::from(16 * 17),
            Fr::from(16 * 17),
            Fr::from(16 * 17),
            Fr::from(16 * 17),
        ],
    ));

    let final_sector = builder.add_sector(out_sector - expected_output_shred);

    builder.set_output(&final_sector);

    let mut circuit = builder.build().unwrap();

    circuit.set_input("Input 1", input_shred_1_data);
    circuit.set_input("Input 2", input_shred_2_data);
    circuit.set_input("Expected Output", expected_output_data);

    let provable_circuit = circuit.finalize().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

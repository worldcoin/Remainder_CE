use super::digits::DigitComponents;
use crate::components::Components;
use crate::digits::{complementary_decomposition, digits_to_field, to_slice_of_mles};
use crate::layouter::builder::{CircuitBuilder, LayerVisibility};
use itertools::Itertools;
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::test_circuit_with_runtime_optimized_config;
use shared_types::Fr;

#[test]
fn test_complementary_recomposition_vertical() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let values = vec![-3, -2, -1, 0, 1, 2, 3, 4];
    let (digits_raw, bits): (Vec<_>, Vec<_>) = values
        .iter()
        .map(|value| complementary_decomposition::<2, 2>(*value).unwrap())
        .unzip();

    let digits = to_slice_of_mles(digits_raw.iter().map(digits_to_field).collect_vec());

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
    let digits_input_shreds = digits
        .iter()
        .enumerate()
        .map(|(i, mle)| {
            builder.add_input_shred(
                &format!("Digits Input Shred {i}"),
                mle.num_vars(),
                &input_layer,
            )
        })
        .collect_vec();

    let bits_input_shred = builder.add_input_shred("Bits Input Shred", 3, &input_layer);

    let values_input_shred = builder.add_input_shred("Values Input Shred", 3, &input_layer);

    let recomp = DigitComponents::unsigned_recomposition(
        &mut builder,
        &digits_input_shreds.iter().collect_vec(),
        2,
    );
    let comp_checker = DigitComponents::complementary_recomp_check(
        &mut builder,
        &values_input_shred,
        &bits_input_shred,
        &recomp,
        2,
        2,
    );

    builder.set_output(&comp_checker);

    let mut circuit = builder.build_with_layer_combination().unwrap();

    digits.iter().enumerate().for_each(|(i, mle)| {
        circuit.set_input(&format!("Digits Input Shred {i}"), mle.clone());
    });
    circuit.set_input("Bits Input Shred", bits.into());
    circuit.set_input("Values Input Shred", values.into());

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
fn test_unsigned_recomposition() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let base: u64 = 16;
    let num_digits = 2;
    let digits: Vec<MultilinearExtension<Fr>> = vec![
        vec![1, 0, 2, 3].into(), // MSBs
        vec![3, 2, 1, 0].into(), // LSBs
    ];
    assert_eq!(digits.len(), num_digits);
    let expected = vec![19, 2, 33, 48].into();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
    let digits_input_shreds = digits
        .iter()
        .enumerate()
        .map(|(i, _)| builder.add_input_shred(&format!("Digits Input Shred {i}"), 2, &input_layer))
        .collect_vec();
    let expected_input_shred = builder.add_input_shred("Expected Input Shred", 2, &input_layer);

    let recomp = DigitComponents::unsigned_recomposition(
        &mut builder,
        &digits_input_shreds.iter().collect_vec(),
        base,
    );

    let equality_checker = Components::equality_check(&mut builder, &expected_input_shred, &recomp);
    builder.set_output(&equality_checker);

    let mut circuit = builder.build_with_layer_combination().unwrap();

    digits.into_iter().enumerate().for_each(|(i, mle)| {
        circuit.set_input(&format!("Digits Input Shred {i}"), mle);
    });
    circuit.set_input("Expected Input Shred", expected);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
fn test_complementary_recomposition() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let base: u64 = 16;
    let num_digits = 2;
    let base_pow = base.pow(num_digits as u32) as i64;
    let digits: Vec<MultilinearExtension<Fr>> = vec![
        vec![1, 0, 2, 3].into(), // MSBs
        vec![3, 2, 1, 0].into(), // LSBs
    ];
    assert_eq!(digits.len(), num_digits);
    let bits: MultilinearExtension<Fr> = vec![1, 0, 1, 0].into(); // 1 iff strictly positive
    let expected: MultilinearExtension<Fr> = vec![base_pow - 19, -2, base_pow - 33, -48].into();

    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
    let digits_input_shreds = digits
        .iter()
        .enumerate()
        .map(|(i, _)| builder.add_input_shred(&format!("Digits Input Shred {i}"), 2, &input_layer))
        .collect_vec();
    let expected_input_shred = builder.add_input_shred("Expected Input Shred", 2, &input_layer);
    let bits_input_shred = builder.add_input_shred("Bits Input Shred", 2, &input_layer);

    let unsigned_recomp = DigitComponents::unsigned_recomposition(
        &mut builder,
        &digits_input_shreds.iter().collect_vec(),
        base,
    );

    let signed_recomp_checker = DigitComponents::complementary_recomp_check(
        &mut builder,
        &expected_input_shred,
        &bits_input_shred,
        &unsigned_recomp,
        base,
        num_digits,
    );

    builder.set_output(&signed_recomp_checker);

    let mut circuit = builder.build_with_layer_combination().unwrap();

    digits.into_iter().enumerate().for_each(|(i, mle)| {
        circuit.set_input(&format!("Digits Input Shred {i}"), mle);
    });
    circuit.set_input("Expected Input Shred", expected);
    circuit.set_input("Bits Input Shred", bits);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
#[should_panic]
fn test_bits_are_binary_soundness() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let bits = vec![3].into();
    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
    let bits_input_shred = builder.add_input_shred("Bits Input Shred", 0, &input_layer);
    let component = DigitComponents::bits_are_binary(&mut builder, &bits_input_shred);
    let _output = builder.set_output(&component);

    let mut circuit = builder.build_with_layer_combination().unwrap();

    circuit.set_input("Bits Input Shred", bits);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

#[test]
fn test_bits_are_binary() {
    let mut builder = CircuitBuilder::<Fr>::new();

    let bits = vec![1, 1, 1, 0].into();
    let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
    let bits_input_shred = builder.add_input_shred("Bits Input Shred", 2, &input_layer);
    let component = DigitComponents::bits_are_binary(&mut builder, &bits_input_shred);
    let _output = builder.set_output(&component);

    let mut circuit = builder.build_with_layer_combination().unwrap();

    circuit.set_input("Bits Input Shred", bits);

    let provable_circuit = circuit.gen_provable_circuit().unwrap();

    test_circuit_with_runtime_optimized_config(&provable_circuit);
}

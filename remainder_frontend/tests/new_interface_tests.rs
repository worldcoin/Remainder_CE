#[cfg(test)]
mod tests {

    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::{curves::ConstantRng, Field, Fr};

    use remainder::{
        layer::gate::BinaryOperation,
        mle::evals::MultilinearExtension,
        prover::helpers::test_circuit_with_runtime_optimized_config,
        utils::mle::{get_random_mle, get_random_mle_from_capacity},
    };
    use remainder_frontend::{
        components::Components,
        layouter::builder::{Circuit, CircuitBuilder, LayerVisibility, NodeRef},
    };

    struct TestComponents;

    impl TestComponents {
        fn difference<F: Field>(
            builder_ref: &mut CircuitBuilder<F>,
            input: &NodeRef<F>,
        ) -> NodeRef<F> {
            let zero_output_sector = builder_ref.add_sector(input - input);

            builder_ref.set_output(&zero_output_sector);

            zero_output_sector
        }

        fn equality_check<F: Field>(
            builder_ref: &mut CircuitBuilder<F>,
            lhs: &NodeRef<F>,
            rhs: &NodeRef<F>,
        ) -> NodeRef<F> {
            let equality_checker = Components::equality_check(builder_ref, lhs, rhs);

            builder_ref.set_output(&equality_checker);

            equality_checker
        }
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_identity_gate_test_circuit<F: Field>(num_free_vars: usize) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // All inputs are public
        let public_input_layer_node =
            builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

        // Circuit inputs as semantic "shreds"
        let first_mle_shred =
            builder.add_input_shred("First MLE Shred", num_free_vars, &public_input_layer_node);
        let second_mle_shred = builder.add_input_shred(
            "Second MLE Shred",
            num_free_vars - 1,
            &public_input_layer_node,
        );

        // Create the circuit components
        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << (num_free_vars - 1);
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx));
        });

        let gate_node = builder.add_identity_gate_node(
            &first_mle_shred,
            nonzero_gates,
            num_free_vars - 1,
            None,
        );

        let _component_2 =
            TestComponents::equality_check(&mut builder, &gate_node, &second_mle_shred);

        builder.build_with_layer_combination().unwrap()
    }

    /// A circuit which takes in two MLEs, select the first half of the first MLE
    /// and compute the difference between that and the second MLE.
    /// The second MLE has one less num_var, and is the same as the half of the
    /// first MLE.
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `first_half_mle` - An MLE whose bookkeeping table is the first half of
    /// `mle`.
    #[test]
    fn test_identity_gate_circuit_newmainder() {
        const NUM_FREE_VARS: usize = 2;
        let mut rng = ConstantRng::new(1);

        let first_mle: MultilinearExtension<Fr> = get_random_mle(NUM_FREE_VARS, &mut rng).mle;
        let second_mle =
            MultilinearExtension::new(first_mle.f.iter().take(1 << (NUM_FREE_VARS - 1)).collect());

        // Create circuit description + input helper function
        let mut circuit = build_identity_gate_test_circuit(NUM_FREE_VARS);

        circuit.set_input("First MLE Shred", first_mle);
        circuit.set_input("Second MLE Shred", second_mle);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_uneven_identity_gate_test_circuit<F: Field>(num_free_vars: usize) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // All inputs are public
        let public_input_layer_node =
            builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

        // Circuit inputs as semantic "shreds"
        let first_mle_shred =
            builder.add_input_shred("First MLE", num_free_vars, &public_input_layer_node);
        let mle_one_element_shred =
            builder.add_input_shred("One Element MLE", 0, &public_input_layer_node);

        // Create the circuit components
        let mut nonzero_gates = vec![];
        nonzero_gates.push((0, 1));

        let gate_node = builder.add_identity_gate_node(&first_mle_shred, nonzero_gates, 0, None);

        let _component_2 =
            TestComponents::equality_check(&mut builder, &gate_node, &mle_one_element_shred);

        builder.build_with_layer_combination().unwrap()
    }

    /// A circuit which takes in two MLEs, select the second element of the first MLE
    /// and compute the difference between that and the second MLE.
    /// The second MLE only has one element, the same as the second element of the
    /// first MLE.
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_one_element` - An MLE whose bookkeeping table is the second element of
    /// `mle`.
    #[test]
    fn test_uneven_identity_gate_circuit_newmainder() {
        const NUM_FREE_VARS: usize = 2;
        let mut rng = test_rng();

        let first_mle: MultilinearExtension<Fr> = get_random_mle(NUM_FREE_VARS, &mut rng).mle;
        // Just the second element of `first_mle`
        let mle_one_element = MultilinearExtension::new(first_mle.iter().skip(1).take(1).collect());

        // Create circuit description + input helper function
        let mut circuit = build_uneven_identity_gate_test_circuit(NUM_FREE_VARS);

        circuit.set_input("First MLE", first_mle);
        circuit.set_input("One Element MLE", mle_one_element);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_dataparallel_identity_gate_test_circuit<F: Field>(
        num_dataparallel_vars: usize,
        num_free_vars: usize,
    ) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // All inputs are public
        let public_input_layer_node =
            builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

        // Circuit inputs as semantic "shreds"
        let dataparallel_first_mle_shred = builder.add_input_shred(
            "Dataparallel First MLE",
            num_dataparallel_vars + num_free_vars,
            &public_input_layer_node,
        );
        let dataparallel_second_mle_shred = builder.add_input_shred(
            "Dataparallel Second MLE",
            num_dataparallel_vars + num_free_vars - 1,
            &public_input_layer_node,
        );

        // Create the circuit components
        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << (num_free_vars - 1);
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx));
        });

        let gate_node = builder.add_identity_gate_node(
            &dataparallel_first_mle_shred,
            nonzero_gates,
            num_dataparallel_vars + num_free_vars - 1,
            Some(num_dataparallel_vars),
        );

        let _component_2 = TestComponents::equality_check(
            &mut builder,
            &gate_node,
            &dataparallel_second_mle_shred,
        );

        builder.build_with_layer_combination().unwrap()
    }

    /// Performs a dataparallel version of [test_identity_gate_circuit_newmainder()].
    /// A circuit which takes in two MLEs, select the first half of the first MLE
    /// and compute the difference between that and the second MLE.
    /// The second MLE has one less num_var, and is the same as the half of the
    /// first MLE.
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `first_half_mle` - An MLE whose bookkeeping table is the first half of
    /// `mle`.
    /// These are similar to their counterparts within [test_add_gate_circuit_newmainder()].
    /// Note that they are interpreted to be dataparallel MLEs with
    /// `2^num_dataparallel_vars` copies of smaller MLEs.
    #[test]
    fn test_dataparallel_identity_gate_circuit_newmainder() {
        const NUM_DATAPARALLEL_VARS: usize = 1;
        const NUM_FREE_VARS: usize = 2;
        let mut rng = test_rng();

        // This should be 2^4
        let dataparallel_first_mle: MultilinearExtension<Fr> =
            get_random_mle(NUM_DATAPARALLEL_VARS + NUM_FREE_VARS, &mut rng).mle;

        // we assume the batch vars are in the beginning
        // so the (individual first halves of batched mles) batched
        // is just the first half of the bookkeeping table of the batched mles
        let dataparallel_second_mle = MultilinearExtension::new(
            (0..(1 << NUM_DATAPARALLEL_VARS))
                .flat_map(|idx| {
                    dataparallel_first_mle.to_vec()[(idx * (1 << NUM_FREE_VARS))
                        ..(idx * (1 << NUM_FREE_VARS) + (1 << (NUM_FREE_VARS - 1)))]
                        .to_vec()
                })
                .collect(),
        );

        // Create circuit description + input helper function
        let mut circuit =
            build_dataparallel_identity_gate_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

        circuit.set_input("Dataparallel First MLE", dataparallel_first_mle);
        circuit.set_input("Dataparallel Second MLE", dataparallel_second_mle);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// Performs a dataparallel version of [test_identity_gate_circuit_newmainder()].
    /// Same as [test_dataparallel_identity_gate_circuit_newmainder()], except that
    /// the dataparallel copies are not a power of 2.
    /// A circuit which takes in two MLEs, select the first half of the first MLE
    /// and compute the difference between that and the second MLE.
    /// The second MLE has one less num_var, and is the same as the half of the
    /// first MLE.
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `first_half_mle` - An MLE whose bookkeeping table is the first half of
    /// `mle`.
    /// These are similar to their counterparts within [test_add_gate_circuit_newmainder()].
    /// Note that they are interpreted to be dataparallel MLEs with
    /// `2^num_dataparallel_vars` copies of smaller MLEs.
    #[test]
    fn test_non_power_of_2_dataparallel_identity_gate_circuit_newmainder() {
        const NUM_DATAPARALLEL_VARS: usize = 2;
        const NUM_DATAPARALLEL_COPIES: usize = (1 << (NUM_DATAPARALLEL_VARS - 1)) + 1;
        const NUM_FREE_VARS: usize = 2;
        let mut rng = test_rng();

        // This should be 2^4 - 2, to make it a non-power of 2
        let dataparallel_first_mle: MultilinearExtension<Fr> = get_random_mle_from_capacity(
            (2_u32.pow(NUM_FREE_VARS as u32) * NUM_DATAPARALLEL_COPIES as u32) as usize,
            &mut rng,
        )
        .mle;

        // we assume the batch vars are in the beginning
        // so the (individual first halves of batched mles) batched
        // is just the first half of the bookkeeping table of the batched mles
        let dataparallel_second_mle = MultilinearExtension::new(
            (0..NUM_DATAPARALLEL_COPIES)
                .flat_map(|idx| {
                    dataparallel_first_mle.to_vec()[(idx * (1 << NUM_FREE_VARS))
                        ..(idx * (1 << NUM_FREE_VARS) + (1 << (NUM_FREE_VARS - 1)))]
                        .to_vec()
                })
                .collect(),
        );
        // Create circuit description + input helper function
        let mut circuit =
            build_dataparallel_identity_gate_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

        circuit.set_input("Dataparallel First MLE", dataparallel_first_mle);
        circuit.set_input("Dataparallel Second MLE", dataparallel_second_mle);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_dataparallel_uneven_identity_gate_test_circuit<F: Field>(
        num_dataparallel_vars: usize,
        num_free_vars: usize,
    ) -> Circuit<F> {
        let mut builder = CircuitBuilder::<F>::new();

        // All inputs are public
        let public_input_layer_node =
            builder.add_input_layer("Public Input Layer", LayerVisibility::Public);

        // Circuit inputs as semantic "shreds"
        let dataparallel_first_mle_shred = builder.add_input_shred(
            "Dataparallel First MLE",
            num_dataparallel_vars + num_free_vars,
            &public_input_layer_node,
        );
        let dataparallel_mle_one_element_shred = builder.add_input_shred(
            "Dataparallel One Element MLE",
            num_dataparallel_vars,
            &public_input_layer_node,
        );

        // Create the circuit components
        let mut nonzero_gates = vec![];
        nonzero_gates.push((0, 1));

        let gate_node = builder.add_identity_gate_node(
            &dataparallel_first_mle_shred,
            nonzero_gates,
            num_dataparallel_vars,
            Some(num_dataparallel_vars),
        );

        let _component_2 = TestComponents::equality_check(
            &mut builder,
            &gate_node,
            &dataparallel_mle_one_element_shred,
        );

        builder.build_with_layer_combination().unwrap()
    }

    /// performs a dataparallel version of [test_uneven_identity_gate_circuit_newmainder()].
    ///
    /// ## Arguments
    /// * `mle` - batched MLE with arbitrary bookkeeping table values.
    /// * `mle_one_element` - batched MLE whose bookkeeping table is the second element of
    /// `mle`.
    #[test]
    fn test_dataparallel_uneven_identity_gate_circuit_newmainder() {
        const NUM_DATAPARALLEL_VARS: usize = 2;
        const NUM_FREE_VARS: usize = 2;
        let mut rng = test_rng();

        let dataparallel_first_mle: MultilinearExtension<Fr> =
            get_random_mle(NUM_DATAPARALLEL_VARS + NUM_FREE_VARS, &mut rng).mle;
        // Just all the second elements of `first_mle`
        let dataparallel_mle_one_element = MultilinearExtension::new(
            (0..1 << NUM_DATAPARALLEL_VARS)
                .map(|idx| {
                    dataparallel_first_mle
                        .get((idx * (1 << NUM_FREE_VARS)) + 1)
                        .unwrap()
                })
                .collect(),
        );

        // Create circuit description + input helper function
        let mut circuit = build_dataparallel_uneven_identity_gate_test_circuit(
            NUM_DATAPARALLEL_VARS,
            NUM_FREE_VARS,
        );

        circuit.set_input("Dataparallel First MLE", dataparallel_first_mle);
        circuit.set_input("Dataparallel One Element MLE", dataparallel_mle_one_element);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        // Prove/verify the circuit
        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// A circuit which takes in two MLEs of the same size and adds
    /// the contents, element-wise, to one another.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `neg_mle` - An MLE whose bookkeeping table is the element-wise negation
    ///     of that of `mle`.
    #[test]
    fn test_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_FREE_BITS: usize = 1;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle = MultilinearExtension::new(mle.iter().map(|elem| -elem).collect());

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let mle_input_shred =
            builder.add_input_shred("Input MLE", mle.clone().num_vars(), &input_layer);
        let neg_mle_input_shred =
            builder.add_input_shred("Neg Input MLE", neg_mle.clone().num_vars(), &input_layer);

        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << mle.num_vars();
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let gate_node = builder.add_gate_node(
            &mle_input_shred,
            &neg_mle_input_shred,
            nonzero_gates,
            BinaryOperation::Add,
            None,
        );

        let _component_2 = TestComponents::difference(&mut builder, &gate_node);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Input MLE", mle);
        circuit.set_input("Neg Input MLE", neg_mle);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// A circuit which takes in two MLEs of the same size, and performs a
    /// dataparallel version of [test_add_gate_circuit_newmainder()].
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_dataparallel`, `neg_mle_dataparallel` -
    ///     Similar to their counterparts within [test_add_gate_circuit_newmainder()]. Note that
    ///     these are interpreted to be dataparallel MLEs with
    ///     `2^num_dataparallel_bits` copies of smaller MLEs.
    /// * `num_dataparallel_bits` - Defines the log_2 of the number of circuit copies.
    #[test]
    fn test_dataparallel_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_DATAPARALLEL_BITS: usize = 2;
        const NUM_FREE_BITS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        let mle_dataparallel =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle_dataparallel =
            MultilinearExtension::new(mle_dataparallel.iter().map(|elem| -elem).collect());

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let dataparallel_mle_input_shred = builder.add_input_shred(
            "Dataparallel Input MLE",
            NUM_DATAPARALLEL_BITS + NUM_FREE_BITS,
            &input_layer,
        );
        let dataparallel_neg_mle_input_shred = builder.add_input_shred(
            "Dataparallel Neg Input MLE",
            NUM_DATAPARALLEL_BITS + NUM_FREE_BITS,
            &input_layer,
        );

        let mut nonzero_gates = vec![];
        let table_size = 1 << (NUM_FREE_BITS);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let gate_node = builder.add_gate_node(
            &dataparallel_mle_input_shred,
            &dataparallel_neg_mle_input_shred,
            nonzero_gates,
            BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_BITS),
        );

        let _component_2 = TestComponents::difference(&mut builder, &gate_node);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Dataparallel Input MLE", mle_dataparallel);
        circuit.set_input("Dataparallel Neg Input MLE", neg_mle_dataparallel);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// A circuit which takes in two MLEs of the same size and adds
    /// only the very first element of `mle` with the first of `neg_mle`.
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle` - An MLE with arbitrary bookkeeping table values.
    /// * `neg_mle` - An MLE whose bookkeeping table is the element-wise negation
    ///     of that of `mle`.
    #[test]
    fn test_uneven_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_FREE_BITS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());
        let neg_mle = MultilinearExtension::new(vec![mle.first().neg()]);

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let mle_input_shred = builder.add_input_shred("Input MLE", NUM_FREE_BITS, &input_layer);
        let neg_mle_input_shred = builder.add_input_shred("Neg Input MLE", 0, &input_layer);

        let nonzero_gates = vec![(0, 0, 0)];
        let gate_node = builder.add_gate_node(
            &mle_input_shred,
            &neg_mle_input_shred,
            nonzero_gates,
            BinaryOperation::Add,
            None,
        );

        let _component_2 = TestComponents::difference(&mut builder, &gate_node);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Input MLE", mle);
        circuit.set_input("Neg Input MLE", neg_mle);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    #[test]
    fn test_mul_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_FREE_VARS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_VARS;

        let mle_1 =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let mle_2 =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle_2 = MultilinearExtension::new(mle_2.iter().map(|elem| -elem).collect());

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let mle_1_input_shred = builder.add_input_shred("Input MLE 1", NUM_FREE_VARS, &input_layer);
        let mle_2_input_shred = builder.add_input_shred("Input MLE 2", NUM_FREE_VARS, &input_layer);
        let neg_mle_2_input_shred =
            builder.add_input_shred("Neg Input MLE 2", NUM_FREE_VARS, &input_layer);

        let mut nonzero_gates = vec![];
        let table_size = 1 << NUM_FREE_VARS;

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = builder.add_gate_node(
            &mle_1_input_shred,
            &neg_mle_2_input_shred,
            nonzero_gates.clone(),
            BinaryOperation::Mul,
            None,
        );

        let pos_mul_output = builder.add_gate_node(
            &mle_1_input_shred,
            &mle_2_input_shred,
            nonzero_gates.clone(),
            BinaryOperation::Mul,
            None,
        );

        let add_gate_layer_output = builder.add_gate_node(
            &pos_mul_output,
            &neg_mul_output,
            nonzero_gates,
            BinaryOperation::Add,
            None,
        );

        let _component_2 = TestComponents::difference(&mut builder, &add_gate_layer_output);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Input MLE 1", mle_1);
        circuit.set_input("Input MLE 2", mle_2);
        circuit.set_input("Neg Input MLE 2", neg_mle_2);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    /// A circuit which takes in two MLEs of the same size, and performs a
    /// dataparallel version of [test_uneven_add_gate_circuit_newmainder()].
    ///
    /// The expected output of this circuit is the zero MLE.
    ///
    /// ## Arguments
    /// * `mle_dataparallel`, `neg_mle_dataparallel` -
    ///     Similar to their counterparts within [test_uneven_add_gate_circuit_newmainder()]. Note that
    ///     these are interpreted to be dataparallel MLEs with
    ///     `2^num_dataparallel_bits` copies of smaller MLEs.
    /// * `num_dataparallel_bits` - Defines the log_2 of the number of circuit copies.
    #[test]
    fn test_dataparallel_uneven_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_DATAPARALLEL_BITS: usize = 4;
        const NUM_FREE_BITS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        let mle_dataparallel =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle_dataparallel =
            MultilinearExtension::new(mle_dataparallel.iter().map(|elem| -elem).collect());

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let dataparallel_mle_input_shred = builder.add_input_shred(
            "Dataparallel Input MLE",
            NUM_DATAPARALLEL_BITS + NUM_FREE_BITS,
            &input_layer,
        );
        let dataparallel_neg_mle_input_shred = builder.add_input_shred(
            "Dataparallel Neg Input MLE",
            NUM_DATAPARALLEL_BITS + NUM_FREE_BITS,
            &input_layer,
        );

        let nonzero_gates = vec![(0, 0, 0)];

        let gate_node = builder.add_gate_node(
            &dataparallel_mle_input_shred,
            &dataparallel_neg_mle_input_shred,
            nonzero_gates,
            BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_BITS),
        );

        let _component_2 = TestComponents::difference(&mut builder, &gate_node);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Dataparallel Input MLE", mle_dataparallel);
        circuit.set_input("Dataparallel Neg Input MLE", neg_mle_dataparallel);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }

    #[test]
    fn test_dataparallel_mul_add_gate_circuit_newmainder() {
        let mut builder = CircuitBuilder::<Fr>::new();

        const NUM_DATAPARALLEL_VARS: usize = 2;
        const NUM_FREE_VARS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_VARS + NUM_FREE_VARS);

        let mle_1_dataparallel =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let mle_2_dataparallel =
            MultilinearExtension::new((0..size).map(|_| Fr::from(rng.gen::<u64>())).collect());

        let neg_mle_2_dataparallel =
            MultilinearExtension::new(mle_2_dataparallel.iter().map(|elem| -elem).collect());

        let input_layer = builder.add_input_layer("Input Layer", LayerVisibility::Public);
        let mle_1_input_shred = builder.add_input_shred(
            "Input MLE 1",
            NUM_FREE_VARS + NUM_DATAPARALLEL_VARS,
            &input_layer,
        );
        let mle_2_input_shred = builder.add_input_shred(
            "Input MLE 2",
            NUM_FREE_VARS + NUM_DATAPARALLEL_VARS,
            &input_layer,
        );
        let neg_mle_2_input_shred = builder.add_input_shred(
            "Neg Input MLE 2",
            NUM_FREE_VARS + NUM_DATAPARALLEL_VARS,
            &input_layer,
        );

        let mut nonzero_gates = vec![];
        let table_size = 1 << NUM_FREE_VARS;

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = builder.add_gate_node(
            &mle_1_input_shred,
            &neg_mle_2_input_shred,
            nonzero_gates.clone(),
            BinaryOperation::Mul,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let pos_mul_output = builder.add_gate_node(
            &mle_1_input_shred,
            &mle_2_input_shred,
            nonzero_gates.clone(),
            BinaryOperation::Mul,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let add_gate_layer_output = builder.add_gate_node(
            &pos_mul_output,
            &neg_mul_output,
            nonzero_gates,
            BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let _component_2 = TestComponents::difference(&mut builder, &add_gate_layer_output);

        let mut circuit = builder.build_with_layer_combination().unwrap();

        circuit.set_input("Input MLE 1", mle_1_dataparallel);
        circuit.set_input("Input MLE 2", mle_2_dataparallel);
        circuit.set_input("Neg Input MLE 2", neg_mle_2_dataparallel);

        let provable_circuit = circuit.gen_provable_circuit().unwrap();

        test_circuit_with_runtime_optimized_config(&provable_circuit);
    }
}

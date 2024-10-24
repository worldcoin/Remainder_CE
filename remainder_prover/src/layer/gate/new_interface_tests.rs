use remainder_shared_types::Field;

use crate::{
    components::EqualityChecker,
    layouter::{
        component::Component,
        nodes::{circuit_outputs::OutputNode, sector::Sector, CircuitNode},
    },
};
// ------------------- COPIED FROM `remainder_prover/tests/utils/mod.rs` -------------------
/// A builder which takes the difference of an MLE from itself to return a zero layer.

pub struct DifferenceBuilderComponent<F: Field> {
    pub output_sector: Sector<F>,
    pub output_node: OutputNode,
}

impl<F: Field, N> Component<N> for DifferenceBuilderComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.output_sector.into(), self.output_node.into()]
    }
}

pub struct EqualityCheckerComponent<F: Field> {
    pub equality_checker: EqualityChecker<F>,
    pub output_node: OutputNode,
}

impl<F: Field, N> Component<N> for EqualityCheckerComponent<F>
where
    N: CircuitNode + From<Sector<F>> + From<OutputNode>,
{
    fn yield_nodes(self) -> Vec<N> {
        vec![self.equality_checker.sector.into(), self.output_node.into()]
    }
}

// ------------------- END COPY -------------------
#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use super::{DifferenceBuilderComponent, EqualityCheckerComponent};
    use ark_std::test_rng;
    use itertools::Itertools;
    use rand::Rng;
    use remainder_shared_types::{Field, Fr};

    use crate::{
        components::EqualityChecker,
        layer::LayerId,
        layouter::{
            component::Component,
            nodes::{
                circuit_inputs::{InputLayerNode, InputShred},
                circuit_outputs::OutputNode,
                gate::GateNode,
                identity_gate::IdentityGateNode,
                node_enum::NodeEnum,
                sector::Sector,
                CircuitNode, Context, NodeId,
            },
        },
        mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
        prover::{generate_circuit_description, helpers::test_circuit_new, GKRCircuitDescription},
        utils::mle::get_random_mle,
    };

    impl<F: Field> DifferenceBuilderComponent<F> {
        fn new(ctx: &Context, input: &dyn CircuitNode) -> Self {
            let zero_output_sector = Sector::new(ctx, &[input], |input_vec| {
                assert_eq!(input_vec.len(), 1);
                let input_data = input_vec[0];
                input_data.expr() - input_data.expr()
            });

            let output_node = OutputNode::new_zero(ctx, &zero_output_sector);

            Self {
                output_sector: zero_output_sector,
                output_node,
            }
        }
    }

    impl<F: Field> EqualityCheckerComponent<F> {
        fn new(ctx: &Context, lhs: &dyn CircuitNode, rhs: &dyn CircuitNode) -> Self {
            let equality_checker = EqualityChecker::new(ctx, lhs, rhs);

            let output_node = OutputNode::new_zero(ctx, &equality_checker.sector);

            Self {
                equality_checker,
                output_node,
            }
        }
    }

    /// Struct which allows for easy "semantic" feeding of inputs into the circuit.
    struct IdentityGateTestInputs<F: Field> {
        first_mle: MultilinearExtension<F>,
        second_mle: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_identity_gate_test_circuit<F: Field>(
        num_free_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(IdentityGateTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Create global context manager ---
        let context = Context::new();

        // --- All inputs are public ---
        let public_input_layer_node = InputLayerNode::new(&context, None);

        // --- Circuit inputs as semantic "shreds" ---
        let first_mle_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
        let second_mle_shred =
            InputShred::new(&context, num_free_vars - 1, &public_input_layer_node);

        // --- Save IDs to be used later ---
        let first_mle_id = first_mle_shred.id();
        let second_mle_id = second_mle_shred.id();

        // --- Create the circuit components ---
        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << (num_free_vars - 1);
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx));
        });

        let gate_node = IdentityGateNode::new(&context, &first_mle_shred, nonzero_gates, None);

        let component_2 = EqualityCheckerComponent::new(&context, &gate_node, &second_mle_shred);

        let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            first_mle_shred.into(),
            second_mle_shred.into(),
            gate_node.into(),
        ];
        all_circuit_nodes.extend(component_2.yield_nodes());

        let (circuit_description, convert_input_shreds_to_input_layers, _) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |test_inputs: IdentityGateTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (first_mle_id, test_inputs.first_mle),
                (second_mle_id, test_inputs.second_mle),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
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
        let mut rng = test_rng();

        let first_mle: MultilinearExtension<Fr> = get_random_mle(NUM_FREE_VARS, &mut rng).mle;
        let second_mle =
            MultilinearExtension::new(first_mle.f.iter().take(1 << (NUM_FREE_VARS - 1)).collect());

        // --- Create circuit description + input helper function ---
        let (circuit_description, input_helper_fn) =
            build_identity_gate_test_circuit(NUM_FREE_VARS);

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(IdentityGateTestInputs {
            first_mle,
            second_mle,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
    }

    /// Struct which allows for easy "semantic" feeding of inputs into the circuit.
    struct UnevenIdentityGateTestInputs<F: Field> {
        first_mle: MultilinearExtension<F>,
        mle_one_element: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_uneven_identity_gate_test_circuit<F: Field>(
        num_free_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(UnevenIdentityGateTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Create global context manager ---
        let context = Context::new();

        // --- All inputs are public ---
        let public_input_layer_node = InputLayerNode::new(&context, None);

        // --- Circuit inputs as semantic "shreds" ---
        let first_mle_shred = InputShred::new(&context, num_free_vars, &public_input_layer_node);
        let mle_one_element_shred = InputShred::new(&context, 0, &public_input_layer_node);

        // --- Save IDs to be used later ---
        let first_mle_id = first_mle_shred.id();
        let mle_one_element_id = mle_one_element_shred.id();

        // --- Create the circuit components ---
        let mut nonzero_gates = vec![];
        nonzero_gates.push((0, 1));

        let gate_node = IdentityGateNode::new(&context, &first_mle_shred, nonzero_gates, None);

        let component_2 =
            EqualityCheckerComponent::new(&context, &gate_node, &mle_one_element_shred);

        let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            first_mle_shred.into(),
            mle_one_element_shred.into(),
            gate_node.into(),
        ];
        all_circuit_nodes.extend(component_2.yield_nodes());

        let (circuit_description, convert_input_shreds_to_input_layers, _) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |test_inputs: UnevenIdentityGateTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (first_mle_id, test_inputs.first_mle),
                (mle_one_element_id, test_inputs.mle_one_element),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
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
        // --- Just the second element of `first_mle` ---
        let mle_one_element = MultilinearExtension::new(first_mle.iter().skip(1).take(1).collect());

        // --- Create circuit description + input helper function ---
        let (circuit_description, input_helper_fn) =
            build_uneven_identity_gate_test_circuit(NUM_FREE_VARS);

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(UnevenIdentityGateTestInputs {
            first_mle,
            mle_one_element,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
    }

    /// Struct which allows for easy "semantic" feeding of inputs into the circuit.
    struct DataparallelIdentityGateTestInputs<F: Field> {
        dataparallel_first_mle: MultilinearExtension<F>,
        dataparallel_second_mle: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_dataparallel_identity_gate_test_circuit<F: Field>(
        num_dataparallel_vars: usize,
        num_free_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(DataparallelIdentityGateTestInputs<F>) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Create global context manager ---
        let context = Context::new();

        // --- All inputs are public ---
        let public_input_layer_node = InputLayerNode::new(&context, None);

        // --- Circuit inputs as semantic "shreds" ---
        let dataparallel_first_mle_shred = InputShred::new(
            &context,
            num_dataparallel_vars + num_free_vars,
            &public_input_layer_node,
        );
        let dataparallel_second_mle_shred = InputShred::new(
            &context,
            num_dataparallel_vars + num_free_vars - 1,
            &public_input_layer_node,
        );

        // --- Save IDs to be used later ---
        let dataparallel_first_mle_id = dataparallel_first_mle_shred.id();
        let dataparallel_second_mle_id = dataparallel_second_mle_shred.id();

        // --- Create the circuit components ---
        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << (num_free_vars - 1);
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx));
        });

        let gate_node = IdentityGateNode::new(
            &context,
            &dataparallel_first_mle_shred,
            nonzero_gates,
            Some(num_dataparallel_vars),
        );

        let component_2 =
            EqualityCheckerComponent::new(&context, &gate_node, &dataparallel_second_mle_shred);

        let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            dataparallel_first_mle_shred.into(),
            dataparallel_second_mle_shred.into(),
            gate_node.into(),
        ];
        all_circuit_nodes.extend(component_2.yield_nodes());

        let (circuit_description, convert_input_shreds_to_input_layers, _) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |test_inputs: DataparallelIdentityGateTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (
                    dataparallel_first_mle_id,
                    test_inputs.dataparallel_first_mle,
                ),
                (
                    dataparallel_second_mle_id,
                    test_inputs.dataparallel_second_mle,
                ),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
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
        const NUM_DATAPARALLEL_VARS: usize = 2;
        const NUM_FREE_VARS: usize = 2;
        let mut rng = test_rng();

        // --- This should be 2^4 ---
        let dataparallel_first_mle: MultilinearExtension<Fr> =
            get_random_mle(NUM_DATAPARALLEL_VARS + NUM_FREE_VARS, &mut rng).mle;

        // we assume the batch vars are in the beginning
        // so the (individual first halves of batched mles) batched
        // is just the first half of the bookkeeping table of the batched mles
        let dataparallel_second_mle = MultilinearExtension::new(
            dataparallel_first_mle
                .iter()
                .take(1 << (NUM_DATAPARALLEL_VARS + NUM_FREE_VARS - 1))
                .collect(),
        );

        // --- Create circuit description + input helper function ---
        let (circuit_description, input_helper_fn) =
            build_dataparallel_identity_gate_test_circuit(NUM_DATAPARALLEL_VARS, NUM_FREE_VARS);

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(DataparallelIdentityGateTestInputs {
            dataparallel_first_mle,
            dataparallel_second_mle,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
    }

    /// Struct which allows for easy "semantic" feeding of inputs into the circuit.
    struct DataparallelUnevenIdentityGateTestInputs<F: Field> {
        dataparallel_first_mle: MultilinearExtension<F>,
        dataparallel_mle_one_element: MultilinearExtension<F>,
    }

    /// Creates the [GKRCircuitDescription] and an associated helper input
    /// function allowing for ease of proving.
    fn build_dataparallel_uneven_identity_gate_test_circuit<F: Field>(
        num_dataparallel_vars: usize,
        num_free_vars: usize,
    ) -> (
        GKRCircuitDescription<F>,
        impl Fn(
            DataparallelUnevenIdentityGateTestInputs<F>,
        ) -> HashMap<LayerId, MultilinearExtension<F>>,
    ) {
        // --- Create global context manager ---
        let context = Context::new();

        // --- All inputs are public ---
        let public_input_layer_node = InputLayerNode::new(&context, None);

        // --- Circuit inputs as semantic "shreds" ---
        let dataparallel_first_mle_shred = InputShred::new(
            &context,
            num_dataparallel_vars + num_free_vars,
            &public_input_layer_node,
        );
        let dataparallel_mle_one_element_shred =
            InputShred::new(&context, num_dataparallel_vars, &public_input_layer_node);

        // --- Save IDs to be used later ---
        let dataparallel_first_mle_id = dataparallel_first_mle_shred.id();
        let dataparallel_mle_one_element_id = dataparallel_mle_one_element_shred.id();

        // --- Create the circuit components ---
        let mut nonzero_gates = vec![];
        nonzero_gates.push((0, 1));

        let gate_node = IdentityGateNode::new(
            &context,
            &dataparallel_first_mle_shred,
            nonzero_gates,
            Some(num_dataparallel_vars),
        );

        let component_2 = EqualityCheckerComponent::new(
            &context,
            &gate_node,
            &dataparallel_mle_one_element_shred,
        );

        let mut all_circuit_nodes: Vec<NodeEnum<F>> = vec![
            public_input_layer_node.into(),
            dataparallel_first_mle_shred.into(),
            dataparallel_mle_one_element_shred.into(),
            gate_node.into(),
        ];
        all_circuit_nodes.extend(component_2.yield_nodes());

        let (circuit_description, convert_input_shreds_to_input_layers, _) =
            generate_circuit_description(all_circuit_nodes).unwrap();

        // --- Write closure which allows easy usage of circuit inputs ---
        let circuit_data_fn = move |test_inputs: DataparallelUnevenIdentityGateTestInputs<F>| {
            let input_shred_id_to_data_mapping: HashMap<NodeId, MultilinearExtension<F>> = vec![
                (
                    dataparallel_first_mle_id,
                    test_inputs.dataparallel_first_mle,
                ),
                (
                    dataparallel_mle_one_element_id,
                    test_inputs.dataparallel_mle_one_element,
                ),
            ]
            .into_iter()
            .collect();
            convert_input_shreds_to_input_layers(input_shred_id_to_data_mapping).unwrap()
        };

        (circuit_description, circuit_data_fn)
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
        // --- Just all the second elements of `first_mle` ---
        let dataparallel_mle_one_element = MultilinearExtension::new(
            (0..1 << NUM_DATAPARALLEL_VARS)
                .map(|idx| {
                    dataparallel_first_mle
                        .get(idx + (1 << (NUM_DATAPARALLEL_VARS)))
                        .unwrap()
                })
                .collect(),
        );
        dbg!(&dataparallel_first_mle);
        dbg!(&dataparallel_mle_one_element);

        // --- Create circuit description + input helper function ---
        let (circuit_description, input_helper_fn) =
            build_dataparallel_uneven_identity_gate_test_circuit(
                NUM_DATAPARALLEL_VARS,
                NUM_FREE_VARS,
            );

        // --- Convert input data into circuit inputs which are assignable by prover ---
        let circuit_inputs = input_helper_fn(DataparallelUnevenIdentityGateTestInputs {
            dataparallel_first_mle,
            dataparallel_mle_one_element,
        });

        // --- Specify private input layers (+ description and precommit), if any ---
        let private_input_layers = HashMap::new();

        // --- Prove/verify the circuit ---
        test_circuit_new(&circuit_description, private_input_layers, &circuit_inputs);
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
        const NUM_FREE_BITS: usize = 1;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle = DenseMle::new_from_iter(mle.mle.iter().map(|elem| -elem), LayerId::Input(0));

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let mle_input_shred = InputShred::new(ctx, mle.mle.clone().num_vars(), &input_layer);
        let mle_input_shred_id = mle_input_shred.id();
        let mle_input_shred_data = MultilinearExtension::new(mle.mle.f.iter().collect_vec());
        let neg_mle_input_shred =
            InputShred::new(ctx, neg_mle.mle.clone().num_vars(), &input_layer);
        let neg_mle_input_shred_id = neg_mle_input_shred.id();
        let neg_mle_input_shred_data =
            MultilinearExtension::new(neg_mle.mle.f.iter().collect_vec());

        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << mle_input_shred.get_num_vars();
        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let gate_node = GateNode::new(
            ctx,
            &mle_input_shred,
            &neg_mle_input_shred,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            None,
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &gate_node);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            mle_input_shred.into(),
            neg_mle_input_shred.into(),
            gate_node.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_data, neg_mle_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_input_shred_id, mle_data);
            input_shred_id_to_data.insert(neg_mle_input_shred_id, neg_mle_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_input_shred_data, neg_mle_input_shred_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
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
        const NUM_DATAPARALLEL_BITS: usize = 2;
        const NUM_FREE_BITS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        let mle_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_dataparallel = DenseMle::new_from_iter(
            mle_dataparallel.mle.iter().map(|elem| -elem),
            LayerId::Input(0),
        );
        let mle_data = mle_dataparallel.mle;
        let neg_mle_data = neg_mle_dataparallel.mle;

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let dataparallel_mle_input_shred =
            InputShred::new(ctx, NUM_DATAPARALLEL_BITS + NUM_FREE_BITS, &input_layer);
        let mle_input_id = dataparallel_mle_input_shred.id();
        let dataparallel_neg_mle_input_shred =
            InputShred::new(ctx, NUM_DATAPARALLEL_BITS + NUM_FREE_BITS, &input_layer);
        let neg_input_id = dataparallel_neg_mle_input_shred.id();

        let mut nonzero_gates = vec![];
        let table_size = 1 << (NUM_FREE_BITS);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let gate_node = GateNode::new(
            ctx,
            &dataparallel_mle_input_shred,
            &dataparallel_neg_mle_input_shred,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_BITS),
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &gate_node);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            dataparallel_mle_input_shred.into(),
            dataparallel_neg_mle_input_shred.into(),
            gate_node.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());
        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_data, neg_mle_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_input_id, mle_data);
            input_shred_id_to_data.insert(neg_input_id, neg_mle_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_data, neg_mle_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
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
        const NUM_FREE_BITS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );
        let neg_mle = DenseMle::new_from_raw(vec![mle.first().neg()], LayerId::Input(0));
        let mle_data = mle.mle;
        let neg_mle_data = neg_mle.mle;

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let mle_input_shred = InputShred::new(ctx, NUM_FREE_BITS, &input_layer);
        let mle_input_shred_id = mle_input_shred.id();
        let neg_mle_input_shred = InputShred::new(ctx, 0, &input_layer);
        let neg_mle_input_shred_id = neg_mle_input_shred.id();

        let nonzero_gates = vec![(0, 0, 0)];
        let gate_node = GateNode::new(
            ctx,
            &mle_input_shred,
            &neg_mle_input_shred,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            None,
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &gate_node);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            mle_input_shred.into(),
            neg_mle_input_shred.into(),
            gate_node.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_data, neg_mle_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_input_shred_id, mle_data);
            input_shred_id_to_data.insert(neg_mle_input_shred_id, neg_mle_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_data, neg_mle_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }

    #[test]
    fn test_mul_add_gate_circuit_newmainder() {
        const NUM_FREE_VARS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_VARS;

        let mle_1: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let mle_2: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_2 = DenseMle::new_from_iter(mle_2.iter().map(|elem| -elem), LayerId::Input(0));

        let mle_1_data = mle_1.mle;
        let mle_2_data = mle_2.mle;
        let neg_mle_2_data = neg_mle_2.mle;

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let mle_1_input_shred = InputShred::new(ctx, NUM_FREE_VARS, &input_layer);
        let mle_1_id = mle_1_input_shred.id();
        let mle_2_input_shred = InputShred::new(ctx, NUM_FREE_VARS, &input_layer);
        let mle_2_id = mle_2_input_shred.id();
        let neg_mle_2_input_shred = InputShred::new(ctx, NUM_FREE_VARS, &input_layer);
        let neg_mle_2_id = neg_mle_2_input_shred.id();

        let mut nonzero_gates = vec![];
        let table_size = 1 << NUM_FREE_VARS;

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = GateNode::new(
            ctx,
            &mle_1_input_shred,
            &neg_mle_2_input_shred,
            nonzero_gates.clone(),
            super::super::BinaryOperation::Mul,
            None,
        );

        let pos_mul_output = GateNode::new(
            ctx,
            &mle_1_input_shred,
            &mle_2_input_shred,
            nonzero_gates.clone(),
            super::super::BinaryOperation::Mul,
            None,
        );

        let add_gate_layer_output = GateNode::new(
            ctx,
            &pos_mul_output,
            &neg_mul_output,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            None,
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &add_gate_layer_output);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            mle_1_input_shred.into(),
            mle_2_input_shred.into(),
            neg_mle_2_input_shred.into(),
            neg_mul_output.into(),
            pos_mul_output.into(),
            add_gate_layer_output.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_1_data, mle_2_data, neg_mle_2_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_1_id, mle_1_data);
            input_shred_id_to_data.insert(mle_2_id, mle_2_data);
            input_shred_id_to_data.insert(neg_mle_2_id, neg_mle_2_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_1_data, mle_2_data, neg_mle_2_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
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
        const NUM_DATAPARALLEL_BITS: usize = 4;
        const NUM_FREE_BITS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        let mle_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_dataparallel = DenseMle::new_from_iter(
            mle_dataparallel.mle.iter().map(|elem| -elem),
            LayerId::Input(0),
        );
        let mle_data = mle_dataparallel.mle;
        let neg_mle_data = neg_mle_dataparallel.mle;

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let dataparallel_mle_input_shred =
            InputShred::new(ctx, NUM_DATAPARALLEL_BITS + NUM_FREE_BITS, &input_layer);
        let mle_input_id = dataparallel_mle_input_shred.id();
        let dataparallel_neg_mle_input_shred =
            InputShred::new(ctx, NUM_DATAPARALLEL_BITS + NUM_FREE_BITS, &input_layer);
        let neg_input_id = dataparallel_neg_mle_input_shred.id();

        let nonzero_gates = vec![(0, 0, 0)];

        let gate_node = GateNode::new(
            ctx,
            &dataparallel_mle_input_shred,
            &dataparallel_neg_mle_input_shred,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_BITS),
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &gate_node);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            dataparallel_mle_input_shred.into(),
            dataparallel_neg_mle_input_shred.into(),
            gate_node.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_data, neg_mle_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_input_id, mle_data);
            input_shred_id_to_data.insert(neg_input_id, neg_mle_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_data, neg_mle_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }

    #[test]
    fn test_dataparallel_mul_add_gate_circuit_newmainder() {
        const NUM_DATAPARALLEL_VARS: usize = 2;
        const NUM_FREE_VARS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_VARS + NUM_FREE_VARS);

        let mle_1_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let mle_2_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_2_dataparallel = DenseMle::new_from_iter(
            mle_2_dataparallel.iter().map(|elem| -elem),
            LayerId::Input(0),
        );

        let mle_1_data = mle_1_dataparallel.mle;
        let mle_2_data = mle_2_dataparallel.mle;
        let neg_mle_2_data = neg_mle_2_dataparallel.mle;

        let ctx = &Context::new();
        let input_layer = InputLayerNode::new(ctx, None);
        let mle_1_input_shred =
            InputShred::new(ctx, NUM_FREE_VARS + NUM_DATAPARALLEL_VARS, &input_layer);
        let mle_1_id = mle_1_input_shred.id();
        let mle_2_input_shred =
            InputShred::new(ctx, NUM_FREE_VARS + NUM_DATAPARALLEL_VARS, &input_layer);
        let mle_2_id = mle_2_input_shred.id();
        let neg_mle_2_input_shred =
            InputShred::new(ctx, NUM_FREE_VARS + NUM_DATAPARALLEL_VARS, &input_layer);
        let neg_mle_2_id = neg_mle_2_input_shred.id();

        let mut nonzero_gates = vec![];
        let table_size = 1 << NUM_FREE_VARS;

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = GateNode::new(
            ctx,
            &mle_1_input_shred,
            &neg_mle_2_input_shred,
            nonzero_gates.clone(),
            super::super::BinaryOperation::Mul,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let pos_mul_output = GateNode::new(
            ctx,
            &mle_1_input_shred,
            &mle_2_input_shred,
            nonzero_gates.clone(),
            super::super::BinaryOperation::Mul,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let add_gate_layer_output = GateNode::new(
            ctx,
            &pos_mul_output,
            &neg_mul_output,
            nonzero_gates,
            super::super::BinaryOperation::Add,
            Some(NUM_DATAPARALLEL_VARS),
        );

        let component_2 = DifferenceBuilderComponent::new(ctx, &add_gate_layer_output);

        let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
            input_layer.into(),
            mle_1_input_shred.into(),
            mle_2_input_shred.into(),
            neg_mle_2_input_shred.into(),
            neg_mul_output.into(),
            pos_mul_output.into(),
            add_gate_layer_output.into(),
        ];
        all_nodes.extend(component_2.yield_nodes());

        let (circ_desc, input_builder_from_shred_map, _input_node_id_to_layer_id) =
            generate_circuit_description(all_nodes).unwrap();

        let input_builder = move |(mle_1_data, mle_2_data, neg_mle_2_data): (
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
            MultilinearExtension<Fr>,
        )| {
            let mut input_shred_id_to_data: HashMap<NodeId, MultilinearExtension<Fr>> =
                HashMap::new();
            input_shred_id_to_data.insert(mle_1_id, mle_1_data);
            input_shred_id_to_data.insert(mle_2_id, mle_2_data);
            input_shred_id_to_data.insert(neg_mle_2_id, neg_mle_2_data);
            input_builder_from_shred_map(input_shred_id_to_data).unwrap()
        };

        let inputs = input_builder((mle_1_data, mle_2_data, neg_mle_2_data));
        test_circuit_new(&circ_desc, HashMap::new(), &inputs);
    }
}

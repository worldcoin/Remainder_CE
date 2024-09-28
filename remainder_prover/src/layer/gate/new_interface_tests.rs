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

    use super::{DifferenceBuilderComponent, EqualityCheckerComponent};
    use ark_std::test_rng;
    use rand::Rng;
    use remainder_shared_types::{Field, Fr};

    use crate::{
        components::EqualityChecker,
        layer::LayerId,
        layouter::{
            compiling::LayouterCircuit,
            component::{Component, ComponentSet},
            nodes::{
                circuit_inputs::{
                    InputLayerData, InputLayerNode, InputLayerType, InputShred, InputShredData,
                },
                circuit_outputs::OutputNode,
                gate::GateNode,
                identity_gate::IdentityGateNode,
                node_enum::NodeEnum,
                sector::Sector,
                CircuitNode, Context,
            },
        },
        mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
        prover::helpers::test_circuit,
        utils::get_input_shred_and_data,
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
        const NUM_FREE_BITS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let half_mle = DenseMle::new_from_iter(
            mle.current_mle.get_evals_vector()[..size / 2]
                .into_iter()
                .map(|elem| *elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let mle_input_shred =
                InputShred::new(ctx, mle.current_mle.clone().num_vars(), &input_layer);
            let mle_input_shred_data = InputShredData::new(
                mle_input_shred.id(),
                MultilinearExtension::new(mle.current_mle.get_evals_vector().to_vec()),
            );
            let half_mle_input_shred =
                InputShred::new(ctx, half_mle.current_mle.clone().num_vars(), &input_layer);
            let half_mle_input_shred_data = InputShredData::new(
                half_mle_input_shred.id(),
                MultilinearExtension::new(half_mle.current_mle.get_evals_vector().to_vec()),
            );

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![mle_input_shred_data, half_mle_input_shred_data],
                None,
            );
            let mut nonzero_gates = vec![];
            let total_num_elems = 1 << half_mle_input_shred.get_num_vars();
            (0..total_num_elems).for_each(|idx| {
                nonzero_gates.push((idx, idx));
            });

            let gate_node = IdentityGateNode::new(ctx, &mle_input_shred, nonzero_gates);

            let component_2 = EqualityCheckerComponent::new(ctx, &gate_node, &half_mle_input_shred);

            let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                mle_input_shred.into(),
                half_mle_input_shred.into(),
                gate_node.into(),
            ];
            all_nodes.extend(component_2.yield_nodes());
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
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

        let neg_mle = DenseMle::new_from_iter(
            mle.current_mle
                .get_evals_vector()
                .clone()
                .into_iter()
                .map(|elem| -elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let mle_input_shred =
                InputShred::new(ctx, mle.current_mle.clone().num_vars(), &input_layer);
            let mle_input_shred_data = InputShredData::new(
                mle_input_shred.id(),
                MultilinearExtension::new(mle.current_mle.get_evals_vector().to_vec()),
            );
            let neg_mle_input_shred =
                InputShred::new(ctx, neg_mle.current_mle.clone().num_vars(), &input_layer);
            let neg_mle_input_shred_data = InputShredData::new(
                neg_mle_input_shred.id(),
                MultilinearExtension::new(neg_mle.current_mle.get_evals_vector().to_vec()),
            );

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![mle_input_shred_data, neg_mle_input_shred_data],
                None,
            );
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
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
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
        const NUM_DATAPARALLEL_BITS: usize = 1;
        const NUM_FREE_BITS: usize = 1;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        // --- This should be 2^4 ---
        let mle_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_dataparallel = DenseMle::new_from_iter(
            mle_dataparallel
                .current_mle
                .get_evals_vector()
                .clone()
                .into_iter()
                .map(|elem| -elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let (dataparallel_mle_input_shred, dataparallel_mle_input_shred_data) =
                get_input_shred_and_data(
                    mle_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );

            let (dataparallel_neg_mle_input_shred, dataparallel_neg_mle_input_shred_data) =
                get_input_shred_and_data(
                    neg_mle_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );
            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![
                    dataparallel_mle_input_shred_data,
                    dataparallel_neg_mle_input_shred_data,
                ],
                None,
            );

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
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
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

        let neg_mle =
            DenseMle::new_from_raw(vec![mle.bookkeeping_table()[0].neg()], LayerId::Input(0));

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let (mle_input_shred, mle_input_shred_data) = get_input_shred_and_data(
                mle.current_mle.get_evals_vector().to_vec(),
                ctx,
                &input_layer,
            );
            let (neg_mle_input_shred, neg_mle_input_shred_data) = get_input_shred_and_data(
                neg_mle.current_mle.get_evals_vector().to_vec(),
                ctx,
                &input_layer,
            );
            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![mle_input_shred_data, neg_mle_input_shred_data],
                None,
            );

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
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
    }

    #[test]
    fn test_mul_add_gate_circuit_newmainder() {
        const NUM_FREE_BITS: usize = 4;

        let mut rng = test_rng();
        let size = 1 << NUM_FREE_BITS;

        let mle_1: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let mle_2: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_2 = DenseMle::new_from_iter(
            mle_2.bookkeeping_table().into_iter().map(|elem| -elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let (mle_1_input_shred, mle_1_input_shred_data) = get_input_shred_and_data(
                mle_1.current_mle.get_evals_vector().to_vec(),
                ctx,
                &input_layer,
            );

            let (mle_2_input_shred, mle_2_input_shred_data) = get_input_shred_and_data(
                mle_2.current_mle.get_evals_vector().to_vec(),
                ctx,
                &input_layer,
            );
            let (neg_mle_2_input_shred, neg_mle_2_input_shred_data) = get_input_shred_and_data(
                neg_mle_2.current_mle.get_evals_vector().to_vec(),
                ctx,
                &input_layer,
            );

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![
                    mle_1_input_shred_data,
                    mle_2_input_shred_data,
                    neg_mle_2_input_shred_data,
                ],
                None,
            );

            let mut nonzero_gates = vec![];
            let table_size = 1 << NUM_FREE_BITS;

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
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
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
            mle_dataparallel
                .current_mle
                .get_evals_vector()
                .iter()
                .map(|elem| -elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let (dataparallel_mle_input_shred, dataparallel_mle_input_shred_data) =
                get_input_shred_and_data(
                    mle_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );
            let (dataparallel_neg_mle_input_shred, dataparallel_neg_mle_input_shred_data) =
                get_input_shred_and_data(
                    neg_mle_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![
                    dataparallel_mle_input_shred_data,
                    dataparallel_neg_mle_input_shred_data,
                ],
                None,
            );
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
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
    }

    #[test]
    fn test_dataparallel_mul_add_gate_circuit_newmainder() {
        const NUM_DATAPARALLEL_BITS: usize = 2;
        const NUM_FREE_BITS: usize = 2;

        let mut rng = test_rng();
        let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_FREE_BITS);

        let mle_1_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let mle_2_dataparallel: DenseMle<Fr> = DenseMle::new_from_iter(
            (0..size).map(|_| Fr::from(rng.gen::<u64>())),
            LayerId::Input(0),
        );

        let neg_mle_2_dataparallel = DenseMle::new_from_iter(
            mle_2_dataparallel
                .bookkeeping_table()
                .into_iter()
                .map(|elem| -elem),
            LayerId::Input(0),
        );

        let circuit = LayouterCircuit::new(|ctx| {
            let input_layer = InputLayerNode::new(ctx, None, InputLayerType::PublicInputLayer);
            let (dataparallel_mle_1_input_shred, dataparallel_mle_1_input_shred_data) =
                get_input_shred_and_data(
                    mle_1_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );
            let (dataparallel_mle_2_input_shred, dataparallel_mle_2_input_shred_data) =
                get_input_shred_and_data(
                    mle_2_dataparallel.current_mle.get_evals_vector().to_vec(),
                    ctx,
                    &input_layer,
                );

            let (dataparallel_neg_mle_2_input_shred, dataparallel_neg_mle_2_input_shred_data) =
                get_input_shred_and_data(
                    neg_mle_2_dataparallel
                        .current_mle
                        .get_evals_vector()
                        .to_vec(),
                    ctx,
                    &input_layer,
                );

            let input_layer_data = InputLayerData::new(
                input_layer.id(),
                vec![
                    dataparallel_mle_1_input_shred_data,
                    dataparallel_mle_2_input_shred_data,
                    dataparallel_neg_mle_2_input_shred_data,
                ],
                None,
            );

            let mut nonzero_gates = vec![];
            let table_size = 1 << NUM_FREE_BITS;

            (0..table_size).for_each(|idx| {
                nonzero_gates.push((idx, idx, idx));
            });

            let neg_mul_output = GateNode::new(
                ctx,
                &dataparallel_mle_1_input_shred,
                &dataparallel_neg_mle_2_input_shred,
                nonzero_gates.clone(),
                super::super::BinaryOperation::Mul,
                Some(NUM_DATAPARALLEL_BITS),
            );

            let pos_mul_output = GateNode::new(
                ctx,
                &dataparallel_mle_1_input_shred,
                &dataparallel_mle_2_input_shred,
                nonzero_gates.clone(),
                super::super::BinaryOperation::Mul,
                Some(NUM_DATAPARALLEL_BITS),
            );

            let add_gate_layer_output = GateNode::new(
                ctx,
                &pos_mul_output,
                &neg_mul_output,
                nonzero_gates,
                super::super::BinaryOperation::Add,
                Some(NUM_DATAPARALLEL_BITS),
            );

            let component_2 = DifferenceBuilderComponent::new(ctx, &add_gate_layer_output);

            let mut all_nodes: Vec<NodeEnum<Fr>> = vec![
                input_layer.into(),
                dataparallel_mle_1_input_shred.into(),
                dataparallel_mle_2_input_shred.into(),
                dataparallel_neg_mle_2_input_shred.into(),
                neg_mul_output.into(),
                pos_mul_output.into(),
                add_gate_layer_output.into(),
            ];
            all_nodes.extend(component_2.yield_nodes());
            (
                ComponentSet::<NodeEnum<Fr>>::new_raw(all_nodes),
                vec![input_layer_data],
            )
        });

        test_circuit(circuit, None)
    }
}

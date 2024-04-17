use ark_std::test_rng;
use rand::Rng;
use remainder_shared_types::{FieldExt, Fr};

use crate::{
    layer::{layer_builder::simple_builders::ZeroBuilder, LayerId},
    mle::{dense::DenseMle, Mle, MleRef},
    prover::{
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
        },
        proof_system::DefaultProofSystem,
        GKRCircuit, Layers, Witness,
    },
};

use super::gate::BinaryOperation;

/// A circuit which takes in two MLEs of the same size and adds
/// the contents, element-wise, to one another.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle` - An MLE with arbitrary bookkeeping table values.
/// * `neg_mle` - An MLE whose bookkeeping table is the element-wise negation
///     of that of `mle`.
struct AddGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    neg_mle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for AddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> = vec![&mut self.mle, &mut self.neg_mle];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << self.mle.mle_ref().num_vars();

        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.neg_mle.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> AddGateCircuit<F> {
    fn new(mle: DenseMle<F, F>, neg_mle: DenseMle<F, F>) -> Self {
        assert_eq!(mle.num_iterated_vars(), neg_mle.num_iterated_vars());
        Self { mle, neg_mle }
    }
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
struct UnevenAddGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    neg_mle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for UnevenAddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> = vec![&mut self.mle, &mut self.neg_mle];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let nonzero_gates = vec![(0, 0, 0)];

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.neg_mle.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> UnevenAddGateCircuit<F> {
    fn new(mle: DenseMle<F, F>, neg_mle: DenseMle<F, F>) -> Self {
        Self { mle, neg_mle }
    }
}

/// A circuit which takes in three MLEs of the same size, and performs the
/// following operation:
/// * First, computes the element-wise multiplication between `mle` and `mle_2`.
/// * Next, computes the element-wise multiplication `mle` and `neg_mle_2`.
/// * Finally, computes the difference between the aforementioned MLEs.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1` - An MLE with arbitrary bookkeeping table values.
/// * `mle_2` - An MLE with arbitrary bookkeeping table values.
/// * `neg_mle_2` - An MLE whose bookkeeping table is the element-wise negation
///     of that of `mle_2`.
struct MulAddGateCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for MulAddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> =
            vec![&mut self.mle_1, &mut self.mle_2, &mut self.neg_mle_2];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let total_num_elems = 1 << self.mle_1.mle_ref().num_vars();

        (0..total_num_elems).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let pos_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.mle_2.mle_ref(),
            None,
            BinaryOperation::Mul,
        );

        let neg_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.neg_mle_2.mle_ref(),
            None,
            BinaryOperation::Mul,
        );

        let add_gate_layer_output = layers.add_gate(
            nonzero_gates,
            pos_mul_output.mle_ref(),
            neg_mul_output.mle_ref(),
            None,
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(add_gate_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> MulAddGateCircuit<F> {
    fn new(mle_1: DenseMle<F, F>, mle_2: DenseMle<F, F>, neg_mle_2: DenseMle<F, F>) -> Self {
        assert_eq!(mle_1.num_iterated_vars(), mle_2.num_iterated_vars());
        assert_eq!(mle_2.num_iterated_vars(), neg_mle_2.num_iterated_vars());
        Self {
            mle_1,
            mle_2,
            neg_mle_2,
        }
    }
}

/// A circuit which takes in three MLEs of the same size, and performs a
/// dataparallel version of [MulAddGateCircuit]
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_1_dataparallel`, `mle_2_dataparallel`, `neg_mle_2_dataparallel` -
///     Similar to their counterparts within [MulAddGateCircuit]. Note that
///     these are interpreted to be dataparallel MLEs with
///     `2^num_dataparallel_bits` copies of smaller MLEs.
/// * `num_dataparallel_bits` - Defines the log_2 of the number of circuit copies.
struct DataparallelMulAddGateCircuit<F: FieldExt> {
    mle_1_dataparallel: DenseMle<F, F>,
    mle_2_dataparallel: DenseMle<F, F>,
    neg_mle_2_dataparallel: DenseMle<F, F>,
    num_dataparallel_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for DataparallelMulAddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> = vec![
            &mut self.mle_1_dataparallel,
            &mut self.mle_2_dataparallel,
            &mut self.neg_mle_2_dataparallel,
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size =
            1 << (self.neg_mle_2_dataparallel.mle_ref().num_vars() - self.num_dataparallel_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1_dataparallel.mle_ref(),
            self.neg_mle_2_dataparallel.mle_ref(),
            Some(self.num_dataparallel_bits),
            BinaryOperation::Mul,
        );

        let pos_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1_dataparallel.mle_ref(),
            self.mle_2_dataparallel.mle_ref(),
            Some(self.num_dataparallel_bits),
            BinaryOperation::Mul,
        );

        let add_gate_layer_output = layers.add_gate(
            nonzero_gates,
            pos_mul_output.mle_ref(),
            neg_mul_output.mle_ref(),
            Some(self.num_dataparallel_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(add_gate_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> DataparallelMulAddGateCircuit<F> {
    fn new(
        mle_1_dataparallel: DenseMle<F, F>,
        mle_2_dataparallel: DenseMle<F, F>,
        neg_mle_2_dataparallel: DenseMle<F, F>,
        num_dataparallel_bits: usize,
    ) -> Self {
        // TODO: Add sanitycheck for dataparallel bits
        assert_eq!(
            mle_1_dataparallel.num_iterated_vars(),
            mle_2_dataparallel.num_iterated_vars()
        );
        assert_eq!(
            mle_2_dataparallel.num_iterated_vars(),
            neg_mle_2_dataparallel.num_iterated_vars()
        );
        Self {
            mle_1_dataparallel,
            mle_2_dataparallel,
            neg_mle_2_dataparallel,
            num_dataparallel_bits,
        }
    }
}

/// A circuit which takes in two MLEs of the same size, and performs a
/// dataparallel version of [AddGateCircuit].
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_dataparallel`, `neg_mle_dataparallel` -
///     Similar to their counterparts within [AddGateCircuit]. Note that
///     these are interpreted to be dataparallel MLEs with
///     `2^num_dataparallel_bits` copies of smaller MLEs.
/// * `num_dataparallel_bits` - Defines the log_2 of the number of circuit copies.
struct DataparallelAddGateCircuit<F: FieldExt> {
    mle_dataparallel: DenseMle<F, F>,
    neg_mle_dataparallel: DenseMle<F, F>,
    num_dataparallel_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for DataparallelAddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> =
            vec![&mut self.mle_dataparallel, &mut self.neg_mle_dataparallel];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size =
            1 << (self.neg_mle_dataparallel.mle_ref().num_vars() - self.num_dataparallel_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle_dataparallel.mle_ref(),
            self.neg_mle_dataparallel.mle_ref(),
            Some(self.num_dataparallel_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);
        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> DataparallelAddGateCircuit<F> {
    fn new(
        mle_dataparallel: DenseMle<F, F>,
        neg_mle_dataparallel: DenseMle<F, F>,
        num_dataparallel_bits: usize,
    ) -> Self {
        // TODO: Add sanitycheck for dataparallel bits
        assert_eq!(
            mle_dataparallel.num_iterated_vars(),
            neg_mle_dataparallel.num_iterated_vars()
        );
        Self {
            mle_dataparallel,
            neg_mle_dataparallel,
            num_dataparallel_bits,
        }
    }
}

/// A circuit which takes in two MLEs of the same size, and performs a
/// dataparallel version of [UnevenAddGateCircuit].
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `mle_dataparallel`, `neg_mle_dataparallel` -
///     Similar to their counterparts within [UnevenAddGateCircuit]. Note that
///     these are interpreted to be dataparallel MLEs with
///     `2^num_dataparallel_bits` copies of smaller MLEs.
/// * `num_dataparallel_bits` - Defines the log_2 of the number of circuit copies.
struct DataparallelUnevenAddGateCircuit<F: FieldExt> {
    mle_dataparallel: DenseMle<F, F>,
    neg_mle_dataparallel: DenseMle<F, F>,
    num_dataparallel_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for DataparallelUnevenAddGateCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<&mut dyn Mle<F>> =
            vec![&mut self.mle_dataparallel, &mut self.neg_mle_dataparallel];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        // let table_size =
        //     1 << (self.neg_mle_dataparallel.mle_ref().num_vars() - self.num_dataparallel_bits);

        nonzero_gates.push((0, 0, 0));

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle_dataparallel.mle_ref(),
            self.neg_mle_dataparallel.mle_ref(),
            Some(self.num_dataparallel_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);
        let output_layer_mle = layers.add_gkr(output_layer_builder);

        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}
impl<F: FieldExt> DataparallelUnevenAddGateCircuit<F> {
    fn new(
        mle_dataparallel: DenseMle<F, F>,
        neg_mle_dataparallel: DenseMle<F, F>,
        num_dataparallel_bits: usize,
    ) -> Self {
        // TODO: Add sanitycheck for dataparallel bits
        assert_eq!(
            mle_dataparallel.num_iterated_vars(),
            neg_mle_dataparallel.num_iterated_vars()
        );
        Self {
            mle_dataparallel,
            neg_mle_dataparallel,
            num_dataparallel_bits,
        }
    }
}

#[test]
fn test_add_gate_circuit() {
    const NUM_ITERATED_BITS: usize = 4;

    let mut rng = test_rng();
    let size = 1 << NUM_ITERATED_BITS;

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle = DenseMle::new_from_iter(
        mle.mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: AddGateCircuit<Fr> = AddGateCircuit::new(mle, neg_mle);
    test_circuit(circuit, None);
}

#[test]
fn test_uneven_add_gate_circuit() {
    const NUM_ITERATED_BITS: usize = 4;
    let mut rng = test_rng();
    let size = 1 << NUM_ITERATED_BITS;

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle = DenseMle::new_from_raw(
        vec![mle.mle_ref().bookkeeping_table()[0].neg()],
        LayerId::Input(0),
        None,
    );

    let circuit: UnevenAddGateCircuit<Fr> = UnevenAddGateCircuit::new(mle, neg_mle);
    test_circuit(circuit, None);
}

#[test]
fn test_dataparallel_add_gate_circuit() {
    const NUM_DATAPARALLEL_BITS: usize = 4;
    const NUM_ITERATED_BITS: usize = 4;

    let mut rng = test_rng();
    let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_ITERATED_BITS);

    // --- This should be 2^4 ---
    let mle_dataparallel: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_dataparallel = DenseMle::new_from_iter(
        mle_dataparallel
            .mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );
    let circuit: DataparallelAddGateCircuit<Fr> = DataparallelAddGateCircuit::new(
        mle_dataparallel,
        neg_mle_dataparallel,
        NUM_DATAPARALLEL_BITS,
    );

    test_circuit(circuit, None);
}

#[test]
fn test_dataparallel_uneven_add_gate_circuit() {
    const NUM_DATAPARALLEL_BITS: usize = 4;
    const NUM_ITERATED_BITS: usize = 4;

    let mut rng = test_rng();
    let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_ITERATED_BITS);

    let mle_dataparallel: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_dataparallel = DenseMle::new_from_iter(
        mle_dataparallel
            .mle_ref()
            .current_mle
            .get_evals_vector()
            .iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: DataparallelUnevenAddGateCircuit<Fr> = DataparallelUnevenAddGateCircuit::new(
        mle_dataparallel,
        neg_mle_dataparallel,
        NUM_DATAPARALLEL_BITS,
    );

    test_circuit(circuit, None);
}

#[test]
fn test_dataparallel_mul_add_gate_circuit() {
    const NUM_DATAPARALLEL_BITS: usize = 4;
    const NUM_ITERATED_BITS: usize = 4;

    let mut rng = test_rng();
    let size = 1 << (NUM_DATAPARALLEL_BITS + NUM_ITERATED_BITS);

    let mle_1_dataparallel: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let mle_2_dataparallel: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_2_dataparallel = DenseMle::new_from_iter(
        mle_2_dataparallel
            .mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: DataparallelMulAddGateCircuit<Fr> = DataparallelMulAddGateCircuit::new(
        mle_1_dataparallel,
        mle_2_dataparallel,
        neg_mle_2_dataparallel,
        NUM_DATAPARALLEL_BITS,
    );

    test_circuit(circuit, None);
}

#[test]
fn test_mul_add_gate_circuit() {
    const NUM_ITERATED_BITS: usize = 4;

    let mut rng = test_rng();
    let size = 1 << NUM_ITERATED_BITS;

    let mle_1: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let neg_mle_2 = DenseMle::new_from_iter(
        mle_2
            .mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: MulAddGateCircuit<Fr> = MulAddGateCircuit::new(mle_1, mle_2, neg_mle_2);

    test_circuit(circuit, None);
}

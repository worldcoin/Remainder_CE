use std::path::Path;

use ark_std::test_rng;
use rand::Rng;
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt, Fr};

use crate::{
    layer::{simple_builders::ZeroBuilder, LayerId},
    mle::{dense::DenseMle, Mle, MleRef},
    prover::{
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
};

use super::gate::BinaryOperation;

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
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

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuitUneven<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuitUneven<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![(0, 0, 0)];

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
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

/// Circuit which just subtracts its two halves with gate mle
struct MulAddSimplestGateCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for MulAddSimplestGateCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle_1),
            Box::new(&mut self.mle_2),
            Box::new(&mut self.neg_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle_1.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
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

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

struct SimplestAddMulBatchedGateCircuit<F: FieldExt> {
    mle_1: DenseMle<F, F>,
    mle_2: DenseMle<F, F>,
    neg_mle_2: DenseMle<F, F>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestAddMulBatchedGateCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.mle_1),
            Box::new(&mut self.mle_2),
            Box::new(&mut self.neg_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size = 1 << (self.neg_mle_2.mle_ref().num_vars() - self.batch_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let neg_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.neg_mle_2.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Mul,
        );

        let pos_mul_output = layers.add_gate(
            nonzero_gates.clone(),
            self.mle_1.mle_ref(),
            self.mle_2.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Mul,
        );

        let add_gate_layer_output = layers.add_gate(
            nonzero_gates,
            pos_mul_output.mle_ref(),
            neg_mul_output.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(add_gate_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

/// Circuit which just subtracts its two halves with gate mle
struct SimplestGateCircuitCombined<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestGateCircuitCombined<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let num_vars = 1 << self.mle.mle_ref().num_vars();

        (0..num_vars).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
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

/// Circuit which just subtracts its two halves with batched gate mle
struct SimplestBatchedGateCircuit<F: FieldExt> {
    mle: DenseMle<F, F>,
    negmle: DenseMle<F, F>,
    batch_bits: usize,
}
impl<F: FieldExt> GKRCircuit<F> for SimplestBatchedGateCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        // --- The input layer should just be the concatenation of `mle` and `output_input` ---
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.mle), Box::new(&mut self.negmle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        // --- Create Layers to be added to ---
        let mut layers = Layers::new();

        let mut nonzero_gates = vec![];
        let table_size = 1 << (self.negmle.mle_ref().num_vars() - self.batch_bits);

        (0..table_size).for_each(|idx| {
            nonzero_gates.push((idx, idx, idx));
        });

        let first_layer_output = layers.add_gate(
            nonzero_gates,
            self.mle.mle_ref(),
            self.negmle.mle_ref(),
            Some(self.batch_bits),
            BinaryOperation::Add,
        );

        let output_layer_builder = ZeroBuilder::new(first_layer_output);

        let output_layer_mle = layers.add_gkr(output_layer_builder);

        // (layers, vec![first_layer_output.mle_ref().get_enum()], input_layer)
        Witness {
            layers,
            output_layers: vec![output_layer_mle.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

#[test]
fn ca() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: SimplestGateCircuit<Fr> = SimplestGateCircuit { mle, negmle };

    test_circuit(circuit, Some(Path::new("./gate_proof_optimized.json")));

    // panic!();
}

#[test]
fn test_gkr_gate_simplest_circuit_uneven() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_raw(
        vec![mle.mle_ref().bookkeeping_table()[0].neg()],
        LayerId::Input(0),
        None,
    );

    let circuit: SimplestGateCircuitUneven<Fr> = SimplestGateCircuitUneven { mle, negmle };

    test_circuit(circuit, Some(Path::new("./gate_proof_testing_uneven.json")));

    // panic!();
}

#[test]
fn test_gkr_gate_simplest_circuit_combined() {
    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^2 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .bookkeeping_table()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        None,
    );

    let circuit: SimplestGateCircuitCombined<Fr> = SimplestGateCircuitCombined { mle, negmle };

    test_circuit(circuit, None);

    // panic!();
}

#[test]
fn test_gkr_gate_batched_simplest_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let mut rng = test_rng();
    let size = 1 << 4;

    // --- This should be 2^4 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        // this is the batched bits
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref()
            .current_mle
            .get_evals_vector()
            .clone()
            .into_iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        // this is the batched bits
        None,
    );

    let circuit: SimplestBatchedGateCircuit<Fr> = SimplestBatchedGateCircuit {
        mle,
        negmle,
        batch_bits: 2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof2_optimized.json")),
    );
}

#[test]
fn test_gkr_gate_batched_simplest_circuit_uneven() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let mut rng = test_rng();
    let size = 1 << 4;
    let size2 = 1 << 3;

    // --- This should be 2^4 ---
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| Fr::from(rng.gen::<u64>())),
        LayerId::Input(0),
        // These are NOT the batched bits
        None,
    );

    let negmle = DenseMle::new_from_iter(
        mle.mle_ref().current_mle.get_evals_vector()[0..size2]
            .iter()
            .map(|elem| -elem),
        LayerId::Input(0),
        // These are NOT the batched bits
        None,
    );

    let circuit: SimplestBatchedGateCircuit<Fr> = SimplestBatchedGateCircuit {
        mle,
        negmle,
        batch_bits: 2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof_uneven_optimized.json")),
    );
}

#[test]
fn test_gkr_add_mul_gate_batched_simplest_circuit() {
    // let subscriber = tracing_subscriber::fmt().with_max_level(Level::TRACE).finish();
    // tracing::subscriber::set_global_default(subscriber)
    //     .map_err(|_err| eprintln!("Unable to set global default subscriber"));

    let _rng = test_rng();
    let size = 1 << 2;

    let mle_1: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            // let num = Fr::from(rng.gen::<u64>());

            Fr::one()
        }),
        LayerId::Input(0),
        None,
    );

    let mle_2: DenseMle<Fr, Fr> = DenseMle::new_from_iter(
        (0..size).map(|_| {
            // let num = Fr::from(rng.gen::<u64>());

            Fr::one()
        }),
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
    // let mle: DenseMle<Fr, Tuple2<Fr>> = DenseMle::new_from_iter(
    //     (0..size).map(|idx| (Fr::from(idx + 1), Fr::from(idx + 1)).into()),
    //     LayerId::Input,
    //     None,
    // );

    let circuit: SimplestAddMulBatchedGateCircuit<Fr> = SimplestAddMulBatchedGateCircuit {
        mle_1,
        mle_2,
        neg_mle_2,
        batch_bits: 1,
    };

    test_circuit(
        circuit,
        Some(Path::new("./gate_batch_proof1_optimized.json")),
    );

    // panic!();
}

#[test]
fn test_gkr_add_mul_gate_simplest_circuit() {
    let mut rng = test_rng();
    let size = 1 << 4;

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

    let circuit: MulAddSimplestGateCircuit<Fr> = MulAddSimplestGateCircuit {
        mle_1,
        mle_2,
        neg_mle_2,
    };

    test_circuit(
        circuit,
        Some(Path::new("./mul_gate_simple_proof_optimized.json")),
    );

    // panic!();
}

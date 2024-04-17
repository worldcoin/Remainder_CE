use ark_std::test_rng;
use itertools::Itertools;

use remainder::{
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::{
        layer_builder::{simple_builders::ZeroBuilder, LayerBuilder},
        LayerId,
    },
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
        },
        proof_system::DefaultProofSystem,
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{FieldExt, Fr};

use crate::utils::get_dummy_random_mle;
pub mod utils;

/// A builder which returns the following expression:
/// - sel(`mle_1`, `mle_1`) + `mle_2` * `mle_2`
///
/// The idea is that the last bit in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `prod_mle` - An MLE with arbitrary bookkeeping table values; same size as `sel_mle`.
struct LastBitLinearBuilder<F: FieldExt> {
    sel_mle: DenseMle<F, F>,
    prod_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for LastBitLinearBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.sel_mle.mle_ref()).concat_expr(Expression::mle(self.sel_mle.mle_ref()))
            + Expression::products(vec![self.prod_mle.mle_ref(), self.prod_mle.mle_ref()])
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let sel_bt = self
            .sel_mle
            .mle
            .iter()
            .zip(self.sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![elem_1, elem_2]);

        let mut prod_bt = self
            .prod_mle
            .mle
            .iter()
            .map(|elem| *elem * elem)
            .collect_vec();
        prod_bt.extend(prod_bt.clone());

        let final_bt = sel_bt
            .zip(prod_bt)
            .map(|(elem_1, elem_2)| *elem_1 + elem_2)
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> LastBitLinearBuilder<F> {
    fn new(sel_mle: DenseMle<F, F>, prod_mle: DenseMle<F, F>) -> Self {
        Self { sel_mle, prod_mle }
    }
}

/// A builder which returns the following expression:
/// - sel(`mle_1` * `mle_1`, `mle_1`)
///
/// The idea is that the first bit (selector bit) in this expression is linear.
///
/// ## Arguments
/// * `sel_mle` - An MLE with arbitrary bookkeeping table values.
struct FirstBitLinearBuilder<F: FieldExt> {
    sel_mle: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for FirstBitLinearBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        Expression::mle(self.sel_mle.mle_ref()).concat_expr(Expression::products(vec![
            self.sel_mle.mle_ref(),
            self.sel_mle.mle_ref(),
        ]))
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let final_bt = self
            .sel_mle
            .mle
            .iter()
            .zip(self.sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![*elem_1 * elem_1, *elem_2])
            .collect_vec();

        DenseMle::new_from_raw(final_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> FirstBitLinearBuilder<F> {
    fn new(sel_mle: DenseMle<F, F>) -> Self {
        Self { sel_mle }
    }
}

/// A circuit which does the following:
/// * Layer 0: [LastBitLinearBuilder] with `sel_mle`, `prod_mle`
/// * Layer 1: [FirstBitLinearBuilder] with `sel_mle`
/// * Layer 2: [ZeroBuilder] with the output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// * `sel_mle`, `prod_mle` both MLEs with arbitrary bookkeeping table values, same size.

struct LinearNonLinearCircuit<F: FieldExt> {
    sel_mle: DenseMle<F, F>,
    prod_mle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for LinearNonLinearCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.sel_mle), Box::new(&mut self.prod_mle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let first_layer_builder =
            LastBitLinearBuilder::new(self.sel_mle.clone(), self.prod_mle.clone());
        let first_layer_output = layers.add_gkr(first_layer_builder);

        let second_layer_builder = FirstBitLinearBuilder::new(first_layer_output);
        let second_layer_output = layers.add_gkr(second_layer_builder);

        let zero_builder = ZeroBuilder::new(second_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

impl<F: FieldExt> LinearNonLinearCircuit<F> {
    fn new(sel_mle: DenseMle<F, F>, prod_mle: DenseMle<F, F>) -> Self {
        assert_eq!(sel_mle.num_iterated_vars(), prod_mle.num_iterated_vars());
        Self { sel_mle, prod_mle }
    }
}

#[test]
fn test_nonlinear_sel_circuit_test() {
    const VARS_SEL_SIDE: usize = 2;
    const VARS_PROD_SIDE: usize = VARS_SEL_SIDE;
    let mut rng = test_rng();

    let sel_mle = get_dummy_random_mle(VARS_SEL_SIDE, &mut rng);
    let prod_mle = get_dummy_random_mle(VARS_PROD_SIDE, &mut rng);

    let linear_non_linear_circuit: LinearNonLinearCircuit<Fr> =
        LinearNonLinearCircuit::new(sel_mle, prod_mle);
    test_circuit(linear_non_linear_circuit, None)
}

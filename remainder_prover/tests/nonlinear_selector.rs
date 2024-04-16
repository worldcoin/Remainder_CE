use ark_std::test_rng;

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
mod utils;

/// A builder which returns the following expression:
/// - sel(`left_sel_mle`, `right_sel_mle`)
///   + `right_sum_mle_1` * `right_sum_mle_2`
///
/// The idea is that this builder has one selector bit which is nonlinear.
///
/// ## Arguments
/// * `left_sel_mle` - An MLE with arbitrary bookkeeping table values.
/// * `right_sel_mle` - An MLE with arbitrary bookkeeping table values, same size as `left_sel_mle`.
/// * `right_sum_mle_1`, `right_sum_mle_2` - MLEs with arbitrary bookkeeping table values, same size,
/// one more variable than `right_sel_mle`.

struct NonlinearSelectorBuilder<F: FieldExt> {
    left_sel_mle: DenseMle<F, F>,
    right_sel_mle: DenseMle<F, F>,
    right_sum_mle_1: DenseMle<F, F>,
    right_sum_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for NonlinearSelectorBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let left_sel_side = Expression::mle(self.left_sel_mle.mle_ref());
        let right_sel_side = Expression::mle(self.right_sel_mle.mle_ref());
        let left_sum_side = right_sel_side.concat_expr(left_sel_side);
        let right_sum_side = Expression::products(vec![
            self.right_sum_mle_1.mle_ref(),
            self.right_sum_mle_2.mle_ref(),
        ]);
        left_sum_side + right_sum_side
    }
    fn next_layer(&self, id: LayerId, prefix_bits: Option<Vec<MleIndex<F>>>) -> Self::Successor {
        let right_side_product_bt: Vec<F> = self
            .right_sum_mle_1
            .mle
            .iter()
            .zip(self.right_sum_mle_2.mle.iter())
            .map(|(elem_1, elem_2)| *elem_1 * elem_2)
            .collect();
        let left_side_concat_bt: Vec<F> = self
            .left_sel_mle
            .mle_ref()
            .bookkeeping_table()
            .iter()
            .zip(self.right_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2])
            .collect();
        let sum_bt: Vec<F> = left_side_concat_bt
            .iter()
            .zip(right_side_product_bt)
            .map(|(left_sum, right_sum)| *left_sum + right_sum)
            .collect();
        DenseMle::new_from_raw(sum_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> NonlinearSelectorBuilder<F> {
    fn new(
        left_sel_mle: DenseMle<F, F>,
        right_sel_mle: DenseMle<F, F>,
        right_sum_mle_1: DenseMle<F, F>,
        right_sum_mle_2: DenseMle<F, F>,
    ) -> Self {
        assert_eq!(
            right_sum_mle_1.mle_ref().num_vars(),
            right_sum_mle_2.mle_ref().num_vars()
        );
        assert_eq!(
            left_sel_mle.mle_ref().num_vars(),
            right_sel_mle.mle_ref().num_vars()
        );
        assert_eq!(
            left_sel_mle.mle_ref().num_vars(),
            right_sum_mle_1.mle_ref().num_vars() - 1
        );
        Self {
            left_sel_mle,
            right_sel_mle,
            right_sum_mle_1,
            right_sum_mle_2,
        }
    }
}

/// A circuit which does the following:
/// * Layer 0: [NonlinearSelectorBuilder] with all inputs.
/// * Layer 1: [ZeroBuilder] with output of Layer 0 and itself.
///
/// The expected output of this circuit is the zero MLE.
///
/// ## Arguments
/// See [NonlinearSelectorBuilder].
struct NonlinearSelectorCircuit<F: FieldExt> {
    left_sel_mle: DenseMle<F, F>,
    right_sel_mle: DenseMle<F, F>,
    right_sum_mle_1: DenseMle<F, F>,
    right_sum_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for NonlinearSelectorCircuit<F> {
    type ProofSystem = DefaultProofSystem;

    fn synthesize(&mut self) -> Witness<F, Self::ProofSystem> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.left_sel_mle),
            Box::new(&mut self.right_sel_mle),
            Box::new(&mut self.right_sum_mle_1),
            Box::new(&mut self.right_sum_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F>>()
            .into();

        let mut layers = Layers::new();

        let first_builder = NonlinearSelectorBuilder::new(
            self.left_sel_mle.clone(),
            self.right_sel_mle.clone(),
            self.right_sum_mle_1.clone(),
            self.right_sum_mle_2.clone(),
        );
        let first_layer_output = layers.add_gkr(first_builder);

        let zero_builder = ZeroBuilder::new(first_layer_output);
        let output = layers.add_gkr(zero_builder);

        Witness {
            layers,
            output_layers: vec![output.get_enum()],
            input_layers: vec![input_layer],
        }
    }
}

impl<F: FieldExt> NonlinearSelectorCircuit<F> {
    fn new(
        left_sel_mle: DenseMle<F, F>,
        right_sel_mle: DenseMle<F, F>,
        right_sum_mle_1: DenseMle<F, F>,
        right_sum_mle_2: DenseMle<F, F>,
    ) -> Self {
        assert_eq!(
            left_sel_mle.num_iterated_vars(),
            right_sel_mle.num_iterated_vars()
        );
        assert_eq!(
            right_sum_mle_1.num_iterated_vars(),
            right_sum_mle_2.num_iterated_vars()
        );
        assert_eq!(
            left_sel_mle.num_iterated_vars() + 1,
            right_sum_mle_1.num_iterated_vars()
        );
        Self {
            left_sel_mle,
            right_sel_mle,
            right_sum_mle_1,
            right_sum_mle_2,
        }
    }
}

#[test]
fn test_nonlinear_sel_circuit_test() {
    const VARS_PRODUCT_SIDE: usize = 3;
    const VARS_SEL_SIDE: usize = VARS_PRODUCT_SIDE - 1;
    let mut rng = &mut test_rng();

    let left_sel_mle = get_dummy_random_mle(VARS_SEL_SIDE, &mut rng);
    let right_sel_mle = get_dummy_random_mle(VARS_SEL_SIDE, &mut rng);
    let right_sum_mle_1 = get_dummy_random_mle(VARS_PRODUCT_SIDE, &mut rng);
    let right_sum_mle_2 = get_dummy_random_mle(VARS_PRODUCT_SIDE, &mut rng);

    let non_linear_sel_circuit: NonlinearSelectorCircuit<Fr> = NonlinearSelectorCircuit::new(
        left_sel_mle,
        right_sel_mle,
        right_sum_mle_1,
        right_sum_mle_2,
    );
    test_circuit(non_linear_sel_circuit, None)
}

use ark_std::test_rng;
use itertools::Itertools;
use rand::Rng;
use remainder::{
    expression::{generic_expr::Expression, prover_expr::ProverExpr},
    layer::{simple_builders::ZeroBuilder, LayerBuilder, LayerId},
    mle::{dense::DenseMle, Mle, MleIndex, MleRef},
    prover::{
        helpers::test_circuit,
        input_layer::{
            combine_input_layers::InputLayerBuilder, public_input_layer::PublicInputLayer,
            InputLayer,
        },
        GKRCircuit, Layers, Witness,
    },
};
use remainder_shared_types::{transcript::poseidon_transcript::PoseidonSponge, FieldExt, Fr};

struct NonlinearNestedSelectorBuilder<F: FieldExt> {
    left_inner_sel_mle: DenseMle<F, F>,
    right_inner_sel_mle: DenseMle<F, F>,
    right_outer_sel_mle: DenseMle<F, F>,
    right_sum_mle_1: DenseMle<F, F>,
    right_sum_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> LayerBuilder<F> for NonlinearNestedSelectorBuilder<F> {
    type Successor = DenseMle<F, F>;

    fn build_expression(&self) -> Expression<F, ProverExpr> {
        let left_inner_sel_side = Expression::mle(self.left_inner_sel_mle.mle_ref());
        let right_inner_sel_side = Expression::mle(self.right_inner_sel_mle.mle_ref());
        let left_outer_sel_side = right_inner_sel_side.concat_expr(left_inner_sel_side);
        let left_sum_side =
            Expression::mle(self.right_outer_sel_mle.mle_ref()).concat_expr(left_outer_sel_side);
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
        let left_side_inner_concat_bt: Vec<F> = self
            .left_inner_sel_mle
            .mle
            .iter()
            .zip(self.right_inner_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2])
            .collect();
        let left_side_bt: Vec<F> = left_side_inner_concat_bt
            .iter()
            .zip(self.right_outer_sel_mle.mle.iter())
            .flat_map(|(elem_1, elem_2)| vec![*elem_1, *elem_2])
            .collect();
        let sum_bt: Vec<F> = left_side_bt
            .iter()
            .zip(right_side_product_bt)
            .map(|(left_sum, right_sum)| *left_sum + right_sum)
            .collect();
        DenseMle::new_from_raw(sum_bt, id, prefix_bits)
    }
}
impl<F: FieldExt> NonlinearNestedSelectorBuilder<F> {
    fn new(
        left_inner_sel_mle: DenseMle<F, F>,
        right_inner_sel_mle: DenseMle<F, F>,
        right_outer_sel_mle: DenseMle<F, F>,
        right_sum_mle_1: DenseMle<F, F>,
        right_sum_mle_2: DenseMle<F, F>,
    ) -> Self {
        assert_eq!(
            right_sum_mle_1.mle_ref().num_vars(),
            right_sum_mle_2.mle_ref().num_vars()
        );
        assert_eq!(
            left_inner_sel_mle.mle_ref().num_vars(),
            right_inner_sel_mle.mle_ref().num_vars()
        );
        assert_eq!(
            left_inner_sel_mle.mle_ref().num_vars(),
            right_outer_sel_mle.mle_ref().num_vars() - 1
        );
        assert_eq!(
            right_outer_sel_mle.mle_ref().num_vars(),
            right_sum_mle_1.mle_ref().num_vars() - 1
        );
        Self {
            left_inner_sel_mle,
            right_inner_sel_mle,
            right_outer_sel_mle,
            right_sum_mle_1,
            right_sum_mle_2,
        }
    }
}

struct NonlinearNestedSelectorCircuit<F: FieldExt> {
    left_inner_sel_mle: DenseMle<F, F>,
    right_inner_sel_mle: DenseMle<F, F>,
    right_outer_sel_mle: DenseMle<F, F>,
    right_sum_mle_1: DenseMle<F, F>,
    right_sum_mle_2: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for NonlinearNestedSelectorCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> = vec![
            Box::new(&mut self.left_inner_sel_mle),
            Box::new(&mut self.right_inner_sel_mle),
            Box::new(&mut self.right_outer_sel_mle),
            Box::new(&mut self.right_sum_mle_1),
            Box::new(&mut self.right_sum_mle_2),
        ];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

        let mut layers = Layers::new();

        let first_builder = NonlinearNestedSelectorBuilder::new(
            self.left_inner_sel_mle.clone(),
            self.right_inner_sel_mle.clone(),
            self.right_outer_sel_mle.clone(),
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

#[test]
fn test_nonlinear_sel_circuit_test() {
    let mut rng = test_rng();
    const VARS_PRODUCT_SIDE: usize = 5;
    const VARS_OUTER_SEL_SIDE: usize = VARS_PRODUCT_SIDE - 1;
    const VARS_INNER_SEL_SIDE: usize = VARS_OUTER_SEL_SIDE - 1;

    let left_inner_sel_mle_vec = (0..(1 << VARS_INNER_SEL_SIDE))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    let right_inner_sel_mle_vec = (0..(1 << VARS_INNER_SEL_SIDE))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    let right_outer_sel_mle_vec = (0..(1 << VARS_OUTER_SEL_SIDE))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    let right_sum_mle_1_vec = (0..(1 << VARS_PRODUCT_SIDE))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();
    let right_sum_mle_2_vec = (0..(1 << VARS_PRODUCT_SIDE))
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect_vec();

    let left_inner_sel_mle =
        DenseMle::new_from_raw(left_inner_sel_mle_vec, LayerId::Input(0), None);
    let right_inner_sel_mle =
        DenseMle::new_from_raw(right_inner_sel_mle_vec, LayerId::Input(0), None);
    let right_outer_sel_mle =
        DenseMle::new_from_raw(right_outer_sel_mle_vec, LayerId::Input(0), None);
    let right_sum_mle_1 = DenseMle::new_from_raw(right_sum_mle_1_vec, LayerId::Input(0), None);
    let right_sum_mle_2 = DenseMle::new_from_raw(right_sum_mle_2_vec, LayerId::Input(0), None);

    let non_linear_sel_circuit: NonlinearNestedSelectorCircuit<Fr> =
        NonlinearNestedSelectorCircuit {
            left_inner_sel_mle,
            right_inner_sel_mle,
            right_outer_sel_mle,
            right_sum_mle_1,
            right_sum_mle_2,
        };
    test_circuit(non_linear_sel_circuit, None)
}

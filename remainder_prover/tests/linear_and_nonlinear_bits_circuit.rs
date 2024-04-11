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

use crate::utils::get_dummy_random_mle;
mod utils;

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

struct LinearNonLinearCircuit<F: FieldExt> {
    sel_mle: DenseMle<F, F>,
    prod_mle: DenseMle<F, F>,
}
impl<F: FieldExt> GKRCircuit<F> for LinearNonLinearCircuit<F> {
    type Sponge = PoseidonSponge<F>;

    fn synthesize(&mut self) -> Witness<F, Self::Sponge> {
        let input_mles: Vec<Box<&mut dyn Mle<F>>> =
            vec![Box::new(&mut self.sel_mle), Box::new(&mut self.prod_mle)];
        let input_layer = InputLayerBuilder::new(input_mles, None, LayerId::Input(0))
            .to_input_layer::<PublicInputLayer<F, _>>()
            .to_enum();

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

#[test]
fn test_nonlinear_sel_circuit_test() {
    const VARS_SEL_SIDE: usize = 2;
    const VARS_PROD_SIDE: usize = VARS_SEL_SIDE;

    let sel_mle = get_dummy_random_mle(VARS_SEL_SIDE);
    let prod_mle = get_dummy_random_mle(VARS_PROD_SIDE);

    let linear_non_linear_circuit: LinearNonLinearCircuit<Fr> = LinearNonLinearCircuit {
        sel_mle: sel_mle,
        prod_mle: prod_mle,
    };
    test_circuit(linear_non_linear_circuit, None)
}

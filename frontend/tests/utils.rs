use ark_std::log2;
use frontend::layouter::{
    builder::{CircuitBuilder, NodeRef},
    nodes::{
        circuit_inputs::{InputLayerNode, InputShred, InputShredData},
        CircuitNode,
    },
};
use itertools::{repeat_n, Itertools};
use rand::Rng;
use remainder::layer::LayerId;
use remainder::mle::evals::{Evaluations, MultilinearExtension};

use frontend::sel_expr;
use remainder::mle::dense::DenseMle;
use remainder::mle::MleIndex;
use shared_types::{Field, Fr, IntoVecF};

/// Returns an MLE with all Fr::one() for testing according to the number of variables.
pub fn get_dummy_one_mle(num_vars: usize) -> DenseMle<Fr> {
    let mle_vec = (0..(1 << num_vars)).map(|_| Fr::one()).collect_vec();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
}

/// Returns an MLE with random elements generated from u64 for testing according to the number of variables.
pub fn get_dummy_random_mle(num_vars: usize, rng: &mut impl Rng) -> DenseMle<Fr> {
    let mle_vec = (0..(1 << num_vars))
        .map(|_| rng.gen::<u64>())
        .collect_vec()
        .into_vec_f();
    DenseMle::new_from_raw(mle_vec, LayerId::Input(0))
}

/// Returns an [InputShred] with the appropriate [MultilinearExtension] as the data generated from random u64
pub fn get_dummy_input_shred_and_data(
    num_vars: usize,
    rng: &mut impl Rng,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<Fr>) {
    let mle_vec = (0..(1 << num_vars))
        .map(|_| rng.gen::<u64>())
        .collect_vec()
        .into_vec_f();
    let data = MultilinearExtension::new_from_evals(Evaluations::new(num_vars, mle_vec));
    let input_shred = InputShred::new(data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Returns an [InputShred] with the appropriate [MultilinearExtension], but given as input an mle_vec
pub fn get_input_shred_and_data_from_vec(
    mle_vec: Vec<Fr>,
    input_node: &InputLayerNode,
) -> (InputShred, InputShredData<Fr>) {
    assert!(mle_vec.len().is_power_of_two());
    let data = MultilinearExtension::new(mle_vec);
    let input_shred = InputShred::new(data.num_vars(), input_node);
    let input_shred_data = InputShredData::new(input_shred.id(), data);
    (input_shred, input_shred_data)
}

/// Returns the total MLE indices given a Vec<bool>
/// for the prefix bits and then the number of free
/// bits after.
pub fn get_total_mle_indices<F: Field>(
    prefix_bits: &[bool],
    num_free_bits: usize,
) -> Vec<MleIndex<F>> {
    prefix_bits
        .iter()
        .map(|bit| MleIndex::Fixed(*bit))
        .chain(repeat_n(MleIndex::Free, num_free_bits))
        .collect()
}

/// Returns the total number of variables which would be present in an MLE
/// which combines the bookkeeping tables of the given MLEs (given in terms
/// of the number of variables representing each one)
pub fn get_total_combined_mle_num_vars(all_num_vars: &[usize]) -> usize {
    log2(all_num_vars.iter().fold(0, |acc, elem| acc + (1 << *elem))) as usize
}

pub struct TestUtilComponents;

impl TestUtilComponents {
    /// A builder which returns an expression with three nested selectors:
    /// * innermost_selector: sel(`inner_inner_sel_mle`, `inner_inner_sel_mle * inner_inner_sel_mle`)
    /// * inner_selector: sel(`innermost_selector`, `inner_sel_mle`)
    /// * overall_expression: sel(`inner_selector`, `outer_sel_mle`).
    ///
    /// ## Arguments
    /// * `inner_inner_sel_mle` - An MLE with arbitrary bookkeeping table values.
    /// * `inner_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
    /// the size of `inner_inner_sel_mle`
    /// * `outer_sel_mle` - An MLE with arbitrary bookkeeping table values, but double
    /// the size of `inner_sel_mle`
    pub fn triple_nested_selector<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        inner_inner_sel: &NodeRef<F>,
        inner_sel: &NodeRef<F>,
        outer_sel: &NodeRef<F>,
    ) -> NodeRef<F> {
        let inner_inner_sel = sel_expr!(inner_inner_sel, inner_inner_sel * inner_inner_sel);
        let inner_sel = sel_expr!(inner_inner_sel, inner_sel);
        let sector_expr = sel_expr!(inner_sel, outer_sel);

        builder_ref.add_sector(sector_expr)
    }

    /// A builder which takes the difference of an MLE from itself to return a zero layer.
    pub fn difference<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        input: &NodeRef<F>,
    ) -> NodeRef<F> {
        let zero_output_sector = builder_ref.add_sector(input - input);

        builder_ref.set_output(&zero_output_sector);

        zero_output_sector
    }

    /// A builder which returns the following expression:
    /// `mle_1 * mle_2 + (10 * mle_1)`
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
    pub fn product_scaled<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1: &NodeRef<F>,
        mle_2: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(mle_1 * mle_2 + mle_1 * F::from(10_u64))
    }

    /// A builder which returns the following expression:
    /// `mle_1 * mle_2 + (mle_1 + mle_2)`.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
    pub fn product_sum<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1: &NodeRef<F>,
        mle_2: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(mle_1 * mle_2 + mle_1 + mle_2)
    }

    /// A builder which returns the following expression:
    /// `mle_1 + 10 + (mle_2 * 10)`.
    ///
    /// ## Arguments
    /// * `mle_1` - An MLE with arbitrary bookkeeping table values.
    /// * `mle_2` - An MLE with arbitrary bookkeeping table values; same size as `mle_1`.
    pub fn constant_scaled_sum<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        mle_1: &NodeRef<F>,
        mle_2: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(mle_1 + F::from(10) + mle_2 * F::from(10))
    }
}

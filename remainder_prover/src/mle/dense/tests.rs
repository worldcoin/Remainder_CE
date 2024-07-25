use crate::mle::evals::DimInfo;
use crate::{mle::dense::DenseMle, utils::get_dummy_random_mle_vec};

use super::*;
use ark_std::test_rng;
use ndarray::{Array, IxDyn};
use remainder_shared_types::Fr;

// ======== `dim_info` tests ========

#[test]
fn create_mle_dim() {
    let dims = IxDyn(&[4, 2, 2]);
    let axes_name = ["data", "tree batch", "sample batch"]
        .map(String::from)
        .to_vec();
    let dim_info = DimInfo::new(dims, axes_name);
    assert!(dim_info.is_ok())
}

#[test]
fn create_mle_dim_mismatch_dims() {
    let dims = IxDyn(&[4, 2, 2, 5]);
    let dims_name = ["data", "tree batch", "sample batch"]
        .map(String::from)
        .to_vec();
    let dim_info = DimInfo::new(dims, dims_name);
    assert!(dim_info.is_err())
}

#[test]
fn set_mle_dim() {
    const NUM_VARS: usize = 5;
    const NUM_DATA_PARALLEL_BITS: usize = 4;
    let mut rng = test_rng();

    let mles: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS, NUM_DATA_PARALLEL_BITS, &mut rng);

    let mle_as_vec = DenseMle::batch_mles(mles).get_padded_evaluations();
    let ndarray_expected = Array::from_shape_vec(
        IxDyn(&[1 << NUM_VARS, 1 << NUM_DATA_PARALLEL_BITS]),
        mle_as_vec.clone(),
    )
    .unwrap();
    let evals = Evaluations::new(NUM_VARS + NUM_DATA_PARALLEL_BITS, mle_as_vec);

    let dims = IxDyn(&[1 << NUM_VARS, 1 << NUM_DATA_PARALLEL_BITS]);
    let axes_name = ["data", "tree batch"].map(String::from).to_vec();
    let dim_info = DimInfo::new(dims, axes_name.clone()).unwrap();

    let mut mle = MultilinearExtension::new_from_evals(evals);
    assert!(mle.set_dim_info(dim_info).is_ok());

    assert_eq!(mle.get_axes_names().unwrap(), axes_name);

    assert_eq!(mle.get_mle_as_ndarray().unwrap(), ndarray_expected);
}

#[test]
fn set_mle_zkdt_dim() {
    const NUM_VARS: usize = 3;
    const TREE_BATCH_NUM_VAR: usize = 4;
    const SAMPLE_BATCH_SIZE_NUM_VAR: usize = 5;
    let mut rng = test_rng();

    let mles: Vec<DenseMle<Fr>> = get_dummy_random_mle_vec(NUM_VARS, TREE_BATCH_NUM_VAR, &mut rng);

    let mle_as_vec = DenseMle::batch_mles(mles).get_padded_evaluations();
    let mle_as_vec: Vec<Fr> = repeat_n(mle_as_vec.clone(), 1 << SAMPLE_BATCH_SIZE_NUM_VAR)
        .flatten()
        .collect();

    let ndarray_expected = Array::from_shape_vec(
        IxDyn(&[
            1 << NUM_VARS,
            1 << TREE_BATCH_NUM_VAR,
            1 << SAMPLE_BATCH_SIZE_NUM_VAR,
        ]),
        mle_as_vec.clone(),
    )
    .unwrap();

    let evals = Evaluations::new(
        NUM_VARS + TREE_BATCH_NUM_VAR + SAMPLE_BATCH_SIZE_NUM_VAR,
        mle_as_vec,
    );

    let dims = IxDyn(&[
        1 << NUM_VARS,
        1 << TREE_BATCH_NUM_VAR,
        1 << SAMPLE_BATCH_SIZE_NUM_VAR,
    ]);
    let axes_name = ["data", "tree batch", "sample batch"]
        .map(String::from)
        .to_vec();
    let dim_info = DimInfo::new(dims, axes_name.clone()).unwrap();

    let mut mle = MultilinearExtension::new_from_evals(evals);
    assert!(mle.set_dim_info(dim_info).is_ok());

    assert_eq!(mle.get_axes_names().unwrap(), axes_name);

    assert_eq!(mle.get_mle_as_ndarray().unwrap(), ndarray_expected);
}

#[test]
fn mle_zkdt_dim_mismatch_with_num_var() {
    const NUM_VARS: usize = 3;
    const TREE_BATCH_NUM_VAR: usize = 4;
    const SAMPLE_BATCH_SIZE_NUM_VAR: usize = 5;
    let mut rng = test_rng();

    let mles: Vec<DenseMle<Fr>> = get_dummy_random_mle_vec(NUM_VARS, TREE_BATCH_NUM_VAR, &mut rng);

    let mle_as_vec = DenseMle::batch_mles(mles).get_padded_evaluations();
    let mle_as_vec = repeat_n(mle_as_vec.clone(), SAMPLE_BATCH_SIZE_NUM_VAR)
        .flatten()
        .collect();
    let evals = Evaluations::new(
        NUM_VARS + TREE_BATCH_NUM_VAR + SAMPLE_BATCH_SIZE_NUM_VAR,
        mle_as_vec,
    );

    let dims = IxDyn(&[
        1 << NUM_VARS,
        1 << TREE_BATCH_NUM_VAR,
        1 << (SAMPLE_BATCH_SIZE_NUM_VAR - 1),
    ]);
    let axes_name = ["data", "tree batch", "sample batch"]
        .map(String::from)
        .to_vec();
    let dim_info = DimInfo::new(dims, axes_name).unwrap();

    let mut mle = MultilinearExtension::new_from_evals(evals);
    assert!(mle.set_dim_info(dim_info).is_err())
}

#[test]
fn mle_dim_mismatch_with_num_var() {
    const NUM_VARS: usize = 4;
    const NUM_DATA_PARALLEL_BITS: usize = 4;
    let mut rng = test_rng();

    let mles: Vec<DenseMle<Fr>> =
        get_dummy_random_mle_vec(NUM_VARS, NUM_DATA_PARALLEL_BITS, &mut rng);

    let mle_as_vec = DenseMle::batch_mles(mles).get_padded_evaluations();
    let evals = Evaluations::new(NUM_VARS + NUM_DATA_PARALLEL_BITS, mle_as_vec);

    let dims = IxDyn(&[1 << 5, 1 << 4]);
    let axes_name = ["data", "tree batch"].map(String::from).to_vec();
    let dim_info = DimInfo::new(dims, axes_name).unwrap();

    let mut mle = MultilinearExtension::new_from_evals(evals);
    assert!(mle.set_dim_info(dim_info).is_err())
}

// ======== `fix_variable` tests ========

#[test]
///test fixing variables in an mle with two variables
fn fix_variable_twovars() {
    let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.fix_variable(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));
    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}
#[test]
///test fixing variables in an mle with three variables
fn fix_variable_threevars() {
    let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.fix_variable(1, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));
    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test nested fixing variables in an mle with three variables
fn fix_variable_nested() {
    let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.fix_variable(1, Fr::from(3));
    mle_ref.fix_variable(2, Fr::from(2));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));
    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test nested fixing all the wayyyy
fn fix_variable_full() {
    let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    let _ = mle_ref.index_mle_indices(0);
    mle_ref.fix_variable(0, Fr::from(3));
    mle_ref.fix_variable(1, Fr::from(2));
    mle_ref.fix_variable(2, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(26)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));
    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

// ======== `fix_variable_at_index` tests ========

#[test]
///test fixing variables in an mle with two variables
fn smart_fix_variable_two_vars_forward() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 1.
    mle_ref.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 2nd variable to 1.
    mle_ref.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
fn smart_fix_variable_two_vars_backwards() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 1.
    mle_ref.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(1), Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 1st variable to 1.
    mle_ref.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_123() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_132() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_213() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_231() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]
///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_312() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}
#[test]

///test fixing variables in an mle with three variables
fn smart_fix_variable_three_vars_321() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(2),
        Fr::from(0),
        Fr::from(3),
        Fr::from(1),
        Fr::from(4),
    ];
    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    let mut mle_ref = mle;
    mle_ref.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle_ref.current_mle.get_evals_vector(),
        *mle_exp.current_mle.get_evals_vector()
    );
}

#[test]

// ======== ========

fn create_dense_mle_from_vec() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(1),
        Fr::from(2),
        Fr::from(3),
        Fr::from(4),
        Fr::from(5),
        Fr::from(6),
        Fr::from(7),
    ];

    //DON'T do this normally, it clones the vec, if you have a flat MLE just use
    // Mle::new
    let mle_iter = DenseMle::new_from_iter(mle_vec.clone().into_iter(), LayerId::Input(0));

    let mle_new: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));

    assert!(*mle_iter.current_mle.get_evals_vector() == *mle_new.current_mle.get_evals_vector());
    assert!(
        mle_iter.num_iterated_vars() == 3 && mle_new.num_iterated_vars() == 3,
        "Num vars must be the log_2 of the length of the vector"
    );
}

// moved to circuit_mle.rs
// #[test]
// fn create_dense_tuple_mle_from_vec() {
//     let tuple_vec = vec![
//         vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)],
//         vec![Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)],
//     ];

//     let tuple2_mle = Tuple2Mle::<Fr>::new_from_raw(tuple_vec, LayerId::Input(0));

//     let (first, second): (Vec<Fr>, Vec<_>) = tuple_vec.into_iter().unzip();

//     assert!(
//         tuple2_mle
//             .get_mle_refs()
//             .iter()
//             .map(|mle| mle.get_padded_evaluations())
//             .collect_vec()
//             == [first, second]
//     );
//     assert!(tuple2_mle.combile_mle_refs().num_iterated_vars() == 3);
// }

#[test]
fn create_dense_mle_ref_from_flat_mle() {
    let mle_vec = vec![
        Fr::from(0),
        Fr::from(1),
        Fr::from(2),
        Fr::from(3),
        Fr::from(4),
        Fr::from(5),
        Fr::from(6),
        Fr::from(7),
    ];

    let mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec.clone(), LayerId::Input(0));

    let mle_ref: DenseMle<Fr> = mle;

    assert!(
        mle_ref.mle_indices == vec![MleIndex::Iterated, MleIndex::Iterated, MleIndex::Iterated]
    );
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_vec);
}

// TODO! move this test, should be layouter's job to include prefix bits
// #[test]
// fn create_dense_mle_ref_from_tuple_mle() {
//     let tuple_vec = vec![
//         (Fr::from(0), Fr::from(1)),
//         (Fr::from(2), Fr::from(3)),
//         (Fr::from(4), Fr::from(5)),
//         (Fr::from(6), Fr::from(7)),
//     ];

//     let tuple2_mle = Tuple2Mle::<Fr>::new_from_raw(tuple_vec, LayerId::Input(0));

//     let mles = tuple2_mle.get_mle_refs();
//     assert_eq!(mles.len(), 2);
//     let first = mles[0];
//     let second = mles[1];

//     assert!(
//         first.mle_indices
//             == vec![
//                 MleIndex::Fixed(false),
//                 MleIndex::Iterated,
//                 MleIndex::Iterated
//             ]
//     );
//     assert!(
//         second.mle_indices
//             == vec![
//                 MleIndex::Fixed(true),
//                 MleIndex::Iterated,
//                 MleIndex::Iterated
//             ]
//     );

//     assert!(first.bookkeeping_table() == &[Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)]);
//     assert!(second.bookkeeping_table() == &[Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]);
// }

use crate::mle::dense::DenseMle;
use crate::mle::evals::DimInfo;

use super::*;
use ndarray::IxDyn;
use shared_types::Fr;

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

// ======== `fix_variable` tests ========

#[test]
/// Test `fix_variable` on an MLE with two variables.
fn fix_variable_two_vars() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mut mle = MultilinearExtension::new(mle_vec);
    mle.fix_variable(Fr::from(1));

    let mle_vec_exp = vec![Fr::from(1), Fr::from(3)];

    assert_eq!(mle.to_vec(), mle_vec_exp);
}
#[test]
/// Test `fix_variable` on an MLE with two variables.
fn fix_variable_three_vars() {
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
    let mut mle = MultilinearExtension::new(mle_vec);
    mle.fix_variable(Fr::from(3));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(5), Fr::from(3), Fr::from(8)];
    assert_eq!(mle.to_vec(), mle_vec_exp);
}

#[test]
/// Test iteratively `fix_variable` on an MLE with three variables.
fn fix_variable_nested() {
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
    let mut mle = MultilinearExtension::new(mle_vec);

    mle.fix_variable(Fr::from(3));
    mle.fix_variable(Fr::from(2));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
    assert_eq!(mle.to_vec(), mle_vec_exp);
}

#[test]
/// Test fixing all the variables in an MLE using `fix_variable`.
fn fix_variable_full() {
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
    let mut mle = MultilinearExtension::new(mle_vec);
    mle.fix_variable(Fr::from(3));
    mle.fix_variable(Fr::from(2));
    mle.fix_variable(Fr::from(4));

    let mle_vec_exp = vec![Fr::from(26)];
    assert_eq!(mle.to_vec(), mle_vec_exp);
}

// ======== `fix_variable_at_index` tests ========

#[test]
/// Test `fix_variable_at_index` with two variables going forward.
fn smart_fix_variable_two_vars_forward() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 1st variable to 1.
    mle.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(1), Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 2nd variable to 1.
    mle.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
fn smart_fix_variable_two_vars_backwards() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 2nd variable to 1.
    mle.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 1st variable to 1.
    mle.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
/// Test `fix_variable_at_index` with three variables in the 123 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(5), Fr::from(3), Fr::from(8)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(12), Fr::from(17)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
/// Test `fix_variable_at_index` with three variables in the 132 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(5), Fr::from(3), Fr::from(8)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(25), Fr::from(28)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
/// Test `fix_variable_at_index` with three variables in the 213 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(12), Fr::from(17)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
/// Test `fix_variable_at_index` with three variables in the 231 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(10), Fr::from(19)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

#[test]
/// Test `fix_variable_at_index` with three variables in the 312 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(10), Fr::from(10), Fr::from(15), Fr::from(16)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(25), Fr::from(28)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}
#[test]

/// Test `fix_variable_at_index` with three variables in the 321 permutation.
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
    let mut mle: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0));
    mle.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(10), Fr::from(10), Fr::from(15), Fr::from(16)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 2nd variable to 4.
    mle.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(10), Fr::from(19)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );

    // Fix 1st variable to 3.
    mle.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(37)];
    let mle_exp: DenseMle<Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0));

    assert_eq!(
        *mle.mle.iter().collect::<Vec<_>>(),
        *mle_exp.mle.iter().collect::<Vec<_>>()
    );
}

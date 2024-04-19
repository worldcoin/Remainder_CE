use super::*;
use remainder_shared_types::Fr;

// ======== `fix_variable` tests ========

#[test]
///test fixing variables in an mle with two variables
fn fix_variable_twovars() {
    let _layer_claims = (vec![Fr::from(3), Fr::from(4)], Fr::one());
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.fix_variable(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.fix_variable(1, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.fix_variable(1, Fr::from(3));
    mle_ref.fix_variable(2, Fr::from(2));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(11)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    let _ = mle_ref.index_mle_indices(0);
    mle_ref.fix_variable(0, Fr::from(3));
    mle_ref.fix_variable(1, Fr::from(2));
    mle_ref.fix_variable(2, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(26)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
}

// ======== `fix_variable_at_index` tests ========

#[test]
///test fixing variables in an mle with two variables
fn smart_fix_variable_two_vars_forward() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 1.
    mle_ref.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(2), Fr::from(3)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 2nd variable to 1.
    mle_ref.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
}

#[test]
fn smart_fix_variable_two_vars_backwards() {
    let mle_vec = vec![Fr::from(5), Fr::from(2), Fr::from(1), Fr::from(3)];
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 1.
    mle_ref.fix_variable_at_index(1, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(1), Fr::from(3)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 1st variable to 1.
    mle_ref.fix_variable_at_index(0, Fr::from(1));

    let mle_vec_exp = vec![Fr::from(3)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(6), Fr::from(9), Fr::from(10)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(6), Fr::from(13)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(7)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(21), Fr::from(26)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);
    let mut mle_ref = mle.mle_ref();
    mle_ref.index_mle_indices(0);

    // Fix 3rd variable to 5.
    mle_ref.fix_variable_at_index(2, Fr::from(5));

    let mle_vec_exp = vec![Fr::from(0), Fr::from(7), Fr::from(5), Fr::from(12)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 2nd variable to 4.
    mle_ref.fix_variable_at_index(1, Fr::from(4));

    let mle_vec_exp = vec![Fr::from(20), Fr::from(27)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);

    // Fix 1st variable to 3.
    mle_ref.fix_variable_at_index(0, Fr::from(3));

    let mle_vec_exp = vec![Fr::from(41)];
    let mle_exp: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec_exp, LayerId::Input(0), None);

    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_exp.mle);
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
    let mle_iter = DenseMle::new_from_iter(mle_vec.clone().into_iter(), LayerId::Input(0), None);

    let mle_new: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec, LayerId::Input(0), None);

    assert!(mle_iter.mle == mle_new.mle);
    assert!(
        mle_iter.num_iterated_vars() == 3 && mle_new.num_iterated_vars() == 3,
        "Num vars must be the log_2 of the length of the vector"
    );
}

#[test]
fn create_dense_tuple_mle_from_vec() {
    let tuple_vec = vec![
        (Fr::from(0), Fr::from(1)),
        (Fr::from(2), Fr::from(3)),
        (Fr::from(4), Fr::from(5)),
        (Fr::from(6), Fr::from(7)),
    ];

    let mle = DenseMle::new_from_iter(
        tuple_vec.clone().into_iter().map(Tuple2::from),
        LayerId::Input(0),
        None,
    );

    let (first, second): (Vec<Fr>, Vec<_>) = tuple_vec.into_iter().unzip();

    assert!(mle.mle == [first, second]);
    assert!(mle.num_iterated_vars() == 3);
}

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

    let mle: DenseMle<Fr, Fr> = DenseMle::new_from_raw(mle_vec.clone(), LayerId::Input(0), None);

    let mle_ref: DenseMleRef<Fr> = mle.mle_ref();

    assert!(
        mle_ref.mle_indices == vec![MleIndex::Iterated, MleIndex::Iterated, MleIndex::Iterated]
    );
    assert_eq!(*mle_ref.current_mle.get_evals_vector(), mle_vec);
}

#[test]
fn create_dense_mle_ref_from_tuple_mle() {
    let tuple_vec = vec![
        (Fr::from(0), Fr::from(1)),
        (Fr::from(2), Fr::from(3)),
        (Fr::from(4), Fr::from(5)),
        (Fr::from(6), Fr::from(7)),
    ];

    let mle = DenseMle::new_from_iter(
        tuple_vec.into_iter().map(Tuple2::from),
        LayerId::Input(0),
        None,
    );

    let first = mle.first();
    let second = mle.second();

    assert!(
        first.mle_indices
            == vec![
                MleIndex::Fixed(false),
                MleIndex::Iterated,
                MleIndex::Iterated
            ]
    );
    assert!(
        second.mle_indices
            == vec![
                MleIndex::Fixed(true),
                MleIndex::Iterated,
                MleIndex::Iterated
            ]
    );

    assert!(first.bookkeeping_table() == &[Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(6)]);
    assert!(second.bookkeeping_table() == &[Fr::from(1), Fr::from(3), Fr::from(5), Fr::from(7)]);
}

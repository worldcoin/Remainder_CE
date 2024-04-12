








// a breakdown of the functions that are tested here
// 1. constructors in prover_expr such as products, mle, constant, sum, scaled, and concat_expr
// 2. increment_mle_vec_indices is indirectly tested because of sum, concat_expr, etc.
// 3. evalute_expr (and fix_variable is indirectly tested bc this)
// 4. evaluate_sumcheck is tested in sumcheck/tests.rs (bc it's in compute_sumcheck_message)
// 5. index_mle_indices







#[test]
fn test_constants_eval() {
    let expression1: Expression<Fr, ProverExpr> = Expression::constant(Fr::one());

    let expression2: Expression<Fr, ProverExpr> = Expression::constant(Fr::from(2));

    let expression3 = expression1.clone() + expression2.clone();

    let mut expression = (expression1 - expression2) * Fr::from(2);
    let mut expression_another = expression.clone() + expression3;

    let challenge = vec![Fr::one()];
    let eval = expression.evaluate_expr(challenge.clone());
    assert_eq!(eval.unwrap(), Fr::from(2).neg());

    let eval_another = expression_another.evaluate_expr(challenge);
    assert_eq!(eval_another.unwrap(), Fr::from(1));
}

#[test]
fn test_mle_eval_two_variable() {
    let mle = DenseMle::new_from_raw(
        vec![Fr::from(4), Fr::from(2), Fr::from(5), Fr::from(7)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mut expression = Expression::mle(mle);
    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 2);

    let challenge = vec![Fr::from(2).neg(), Fr::from(9)];
    let eval = expression.evaluate_expr(challenge);
    assert_eq!(eval.unwrap(), Fr::from(55).neg());
}

#[test]
fn test_mle_eval_three_variable() {
    let mle = DenseMle::new_from_raw(
        vec![
            Fr::from(4),
            Fr::from(2),
            Fr::from(5),
            Fr::from(7),
            Fr::from(2),
            Fr::from(4),
            Fr::from(9),
            Fr::from(6),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mut expression = Expression::mle(mle);
    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 3);

    let challenge = vec![Fr::from(2).neg(), Fr::from(3), Fr::from(5)];
    let eval = expression.evaluate_expr(challenge);
    assert_eq!(eval.unwrap(), Fr::from(297));
}

#[test]
fn test_mle_eval_sum_w_constant_then_scale() {
    let mle = DenseMle::new_from_raw(
        vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression = Expression::mle(mle);
    let mut expression = (expression + Expression::constant(Fr::from(5))) * Fr::from(2);
    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 2);

    let challenge = vec![Fr::from(1).neg(), Fr::from(7)];
    let eval = expression.evaluate_expr(challenge);
    assert_eq!(eval.unwrap(), Fr::from(132).neg());
}

#[test]
fn test_mle_eval_selector() {
    let mle_1 = DenseMle::new_from_raw(
        vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_1 = Expression::mle(mle_1);

    let mle_2 = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(9), Fr::from(8), Fr::from(2)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_2 = Expression::mle(mle_2);

    let mut expression = expression_1.concat_expr(expression_2);

    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 3);

    let challenge = vec![Fr::from(2), Fr::from(7), Fr::from(3)];
    let eval = expression.evaluate_expr(challenge);

    let mle_concat = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
            Fr::from(4),
            Fr::from(2),
            Fr::from(1),
            Fr::from(7),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref(); // cancat actually prepends

    let challenge_concat = vec![Fr::from(7), Fr::from(3), Fr::from(2)]; // move the first challenge towards the end

    let mut expression_concat = Expression::mle(mle_concat);

    let num_indices_concat = expression_concat.index_mle_indices(0);
    assert_eq!(num_indices_concat, 3);

    let eval_concat = expression_concat.evaluate_expr(challenge_concat);

    assert_eq!(eval.unwrap(), eval_concat.unwrap());
}

#[test]
fn test_mle_eval_selector_w_constant() {
    let mle_1 = DenseMle::new_from_raw(
        vec![Fr::from(4), Fr::from(2), Fr::from(1), Fr::from(7)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_1 = Expression::mle(mle_1);

    let mut expression = expression_1.concat_expr(Expression::constant(Fr::from(5)));

    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 3);

    let challenge = vec![Fr::from(1).neg(), Fr::from(7), Fr::from(3)];
    let eval = expression.evaluate_expr(challenge);

    // -149 + (1 - (-1)) * 5
    assert_eq!(eval.unwrap(), Fr::from(139).neg());
}

#[test]
fn test_mle_refs_eval() {
    let challenge = vec![Fr::from(2), Fr::from(3)];

    let mle_1 = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mut expression_1 = Expression::mle(mle_1.clone());
    let _ = expression_1.index_mle_indices(0);
    let eval_1 = expression_1.evaluate_expr(challenge.clone()).unwrap();

    let mle_2 = DenseMle::new_from_raw(
        vec![Fr::from(1), Fr::from(4), Fr::from(5), Fr::from(2)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mut expression_2 = Expression::mle(mle_2.clone());
    let _ = expression_2.index_mle_indices(0);
    let eval_2 = expression_2.evaluate_expr(challenge.clone()).unwrap();

    let mut expression_product = Expression::products(vec![mle_1, mle_2]);
    let num_indices = expression_product.index_mle_indices(0);
    assert_eq!(num_indices, 2);

    let eval_prod = expression_product.evaluate_expr(challenge).unwrap();

    assert_eq!(eval_prod, (eval_1 * eval_2));
    // 11 * -17
    assert_eq!(eval_prod, Fr::from(187).neg());
}

#[test]
fn test_mle_different_length_eval() {
    let challenge = vec![Fr::from(2), Fr::from(3), Fr::from(5)];

    let mle_1 = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_1 = Expression::mle(mle_1);

    let mle_2 = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_2 = Expression::mle(mle_2);

    let mut expression = expression_1 + expression_2;
    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 3);

    let eval_prod = expression.evaluate_expr(challenge).unwrap();

    // -230 + 68 + 11
    assert_eq!(eval_prod, Fr::from(151).neg());
}

#[test]
fn test_all_mle_indices() {
    let mle_1: crate::mle::dense::DenseMleRef<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mle_2 = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_product = Box::new(Expression::products(vec![mle_1.clone(), mle_2.clone()]));
    let expression_mle = Expression::mle(mle_2);
    let expression_product_2 = Expression::products(vec![mle_1.clone(), mle_1]);
    let mut expression_full = Expression::sum(
        expression_product,
        Box::new(expression_mle.concat_expr(expression_product_2)),
    );
    expression_full.index_mle_indices(0);
    let mut curr_all_indices: Vec<usize> = Vec::new();
    let all_indices = expression_full
        .expression_node
        .get_all_rounds(&mut curr_all_indices, &expression_full.mle_vec);
    let expected_indices_vec: Vec<usize> = vec![0, 1, 2, 3];
    let expected_all_indices: HashSet<&usize> = HashSet::from_iter(expected_indices_vec.iter());
    let actual_all_indices: HashSet<&usize> = HashSet::from_iter(all_indices.iter());
    assert_eq!(expected_all_indices, actual_all_indices);
}

#[test]
fn test_nonlinear_mle_indices() {
    let mle_1: crate::mle::dense::DenseMleRef<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mle_2 = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_product = Box::new(Expression::products(vec![mle_1.clone(), mle_2.clone()]));
    let expression_mle = Expression::mle(mle_2);
    let expression_product_2 = Expression::products(vec![mle_1.clone(), mle_1]);
    let mut expression_full = Expression::sum(
        expression_product,
        Box::new(expression_mle.concat_expr(expression_product_2)),
    );
    expression_full.index_mle_indices(0);
    let mut curr_all_indices: Vec<usize> = Vec::new();
    let all_nonlinear_indices = expression_full
        .expression_node
        .get_all_nonlinear_rounds(&mut curr_all_indices, &expression_full.mle_vec);
    let expected_nonlinear_indices_vec = vec![0, 1, 2];
    let expected_all_nonlinear_indices: HashSet<&usize> =
        HashSet::from_iter(expected_nonlinear_indices_vec.iter());
    let actual_all_nonlinear_indices: HashSet<&usize> =
        HashSet::from_iter(all_nonlinear_indices.iter());
    assert_eq!(expected_all_nonlinear_indices, actual_all_nonlinear_indices);
}

#[test]
fn test_linear_mle_indices() {
    let mle_1: crate::mle::dense::DenseMleRef<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mle_2 = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_product = Box::new(Expression::products(vec![mle_1.clone(), mle_2.clone()]));
    let expression_mle = Expression::mle(mle_2);
    let expression_product_2 = Expression::products(vec![mle_1.clone(), mle_1]);
    let mut expression_full = Expression::sum(
        expression_product,
        Box::new(expression_mle.concat_expr(expression_product_2)),
    );
    expression_full.index_mle_indices(0);
    let all_linear_indices = expression_full
        .expression_node
        .get_all_linear_rounds(&expression_full.mle_vec);
    let expected_linear_indices_vec = vec![3];
    let expected_all_linear_indices: HashSet<&usize> =
        HashSet::from_iter(expected_linear_indices_vec.iter());
    let actual_all_linear_indices: HashSet<&usize> = HashSet::from_iter(all_linear_indices.iter());
    assert_eq!(expected_all_linear_indices, actual_all_linear_indices);
}

#[test]
fn test_linear_mle_indices_2() {
    let mle_1: crate::mle::dense::DenseMleRef<Fr> = DenseMle::new_from_raw(
        vec![Fr::from(2), Fr::from(2), Fr::from(1), Fr::from(3)],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let mle_2 = DenseMle::new_from_raw(
        vec![
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(2),
            Fr::from(1),
            Fr::from(9),
            Fr::from(8),
            Fr::from(2),
        ],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression_product = Box::new(Expression::products(vec![mle_1.clone(), mle_2.clone()]));
    let expression_mle = Expression::mle(mle_2);
    let expression_product_2 = Expression::products(vec![mle_1.clone(), mle_1]);
    let expression_half = Expression::sum(
        expression_product.clone(),
        Box::new(expression_mle.concat_expr(expression_product_2.clone())),
    );
    let expression_other_half = Expression::sum(
        Box::new(expression_product_2),
        Box::new(Expression::negated(expression_product)),
    );
    let mut expression_full = expression_half.concat_expr(expression_other_half);
    expression_full.index_mle_indices(0);
    let all_linear_indices = expression_full
        .expression_node
        .get_all_linear_rounds(&expression_full.mle_vec);
    let expected_linear_indices_vec = vec![0, 4];
    let expected_all_linear_indices: HashSet<&usize> =
        HashSet::from_iter(expected_linear_indices_vec.iter());
    let actual_all_linear_indices: HashSet<&usize> = HashSet::from_iter(all_linear_indices.iter());
    assert_eq!(expected_all_linear_indices, actual_all_linear_indices);
}

// #[test]
// fn test_not_fully_bounded_eval() {
//     let mle = DenseMle::<_, Fr>::new(vec![
//         Fr::from(4),
//         Fr::from(2),
//         Fr::from(5),
//         Fr::from(7),
//         Fr::from(2),
//         Fr::from(4),
//         Fr::from(9),
//         Fr::from(6),
//     ])
//     .mle_ref();

//     let mut expression = Expression::mle(mle);
//     let _ = expression.index_mle_indices(3);

//     let challenge = vec![Fr::from(-2), Fr::from(3), Fr::from(5)];
//     let eval = expression.evaluate_expr(challenge);
//     assert_eq!(eval, Err(ExpressionError::EvaluateNotFullyBoundError));
// }

#[test]
fn big_test_eval() {
    let expression1: Expression<Fr, ProverExpr> = Expression::constant(Fr::one());

    let mle = DenseMle::new_from_raw(
        vec![Fr::one(), Fr::from(2), Fr::from(3), Fr::one()],
        LayerId::Input(0),
        None,
    )
    .mle_ref();

    let expression3 = Expression::mle(mle.clone());

    let expression = expression1.clone() + expression3.clone();

    let expression_product = Expression::pow(2, mle);

    let expression = expression_product + expression;

    let expression = expression1 - expression;

    let expression = expression * Fr::from(2);

    let mut expression = expression3.concat_expr(expression);
    let num_indices = expression.index_mle_indices(0);
    assert_eq!(num_indices, 3);

    let challenge = vec![Fr::from(2), Fr::from(3), Fr::from(4)];
    let eval = expression.evaluate_expr(challenge).unwrap();

    // -((1 - (24 * 24 - 23)) * 2) - 24 * 2
    assert_eq!(eval, Fr::from(1056));
}

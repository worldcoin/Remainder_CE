use ark_std::{cfg_into_iter, cfg_iter};
use itertools::Itertools;

use std::fmt::Debug;

use crate::{
    mle::{betavalues::BetaValues, Mle},
    sumcheck::*,
    utils::mle::{
        compute_flipped_bit_idx_and_values_lexicographic,
        compute_inverses_vec_and_one_minus_inverted_vec, compute_next_beta_value_from_current,
        compute_next_beta_values_vec_from_current,
    },
};
use remainder_shared_types::Field;

use crate::mle::{dense::DenseMle, MleIndex};
use thiserror::Error;

use super::BinaryOperation;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Error handling for gate mle construction.
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("phase 1 not initialized")]
    /// Error when initializing the first phase, which is when we bind the "x"
    /// bits.
    Phase1InitError,
    #[error("phase 2 not initialized")]
    /// Error when initializing the second phase, which is when we bind the "y"
    /// bits.
    Phase2InitError,
    #[error("mle not fully bound")]
    /// We are on the last round of sumcheck and want to grab claims, but the
    /// MLE is not fully bound which should not be the case for the last round
    /// of sumcheck.
    MleNotFullyBoundError,
    #[error("empty list for lhs or rhs")]
    /// We are initializing a gate on something that does not have either a left
    /// or right side of the expression.
    EmptyMleList,
    #[error("bound indices fail to match challenge")]
    /// When checking the last round of sumcheck, the challenges don't match
    /// what is bound to the MLE.
    EvaluateBoundIndicesDontMatch,
}

/// Given (possibly half-fixed) bookkeeping tables of the MLEs which are
/// multiplied, e.g. V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i +
/// 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) computes g_k(x) = \sum_{b_{k
/// + 1}, ..., b_n} V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i +
///   1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) at `degree + 1` points.
///
/// ## Arguments
/// * `mles` - MLEs pointing to the actual bookkeeping tables for the above
/// * `independent_variable` - whether the `x` from above resides within at
///   least one of the `mles`
/// * `degree` - degree of `g_k(x)`, i.e. number of evaluations to send (minus
///   one!)
pub fn evaluate_mle_product_no_beta_table<F: Field>(
    mles: &[&impl Mle<F>],
    independent_variable: bool,
    degree: usize,
) -> Result<SumcheckEvals<F>, MleError> {
    // Gets the total number of free variables across all MLEs within this
    // product
    let max_num_vars = mles
        .iter()
        .map(|mle| mle.num_free_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    if independent_variable {
        // There is an independent variable, and we must extract `degree`
        // evaluations of it, over `0..degree`
        let eval_count = degree + 1;
        let mle_num_coefficients_mid = 1 << (max_num_vars - 1);

        // iterate across all pairs of evaluations
        let evals = cfg_into_iter!((0..mle_num_coefficients_mid)).fold(
            #[cfg(feature = "parallel")]
            || vec![F::ZERO; eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::ZERO; eval_count],
            |mut acc, index| {
                //get the product of all evaluations over 0/1/..degree
                let evals = mles
                    .iter()
                    .map(|mle| {
                        let index = if mle.num_free_vars() < max_num_vars {
                            let max = 1 << mle.num_free_vars();
                            let multiple = (1 << max_num_vars) / max;
                            index / multiple
                        } else {
                            index
                        };
                        let first = mle.get(index).unwrap_or(F::ZERO);
                        let second = if mle.num_free_vars() != 0 {
                            mle.get(index + mle_num_coefficients_mid).unwrap_or(F::ZERO)
                        } else {
                            first
                        };

                        let step = second - first;

                        let successors =
                            std::iter::successors(Some(second), move |item| Some(*item + step));
                        // iterator that represents all evaluations of the MLE
                        // extended to arbitrarily many linear extrapolations on
                        // the line of 0/1
                        std::iter::once(first).chain(successors)
                    })
                    .map(|item| -> Box<dyn Iterator<Item = F>> { Box::new(item) })
                    .reduce(|acc, evals| Box::new(acc.zip(evals).map(|(acc, eval)| acc * eval)))
                    .unwrap();

                acc.iter_mut()
                    .zip(evals)
                    .for_each(|(acc, eval)| *acc += eval);
                acc
            },
        );

        #[cfg(feature = "parallel")]
        let evals = evals.reduce(
            || vec![F::ZERO; eval_count],
            |mut acc, partial| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );

        Ok(SumcheckEvals(evals))
    } else {
        // There is no independent variable and we can sum over everything
        let sum = cfg_into_iter!((0..(1 << max_num_vars))).fold(
            #[cfg(feature = "parallel")]
            || F::ZERO,
            #[cfg(not(feature = "parallel"))]
            F::ZERO,
            |acc, index| {
                // Go through each MLE within the product
                let product = mles
                    .iter()
                    // Result of this `map()`: A list of evaluations of the MLEs
                    // at `index`
                    .map(|mle| {
                        let index = if mle.num_free_vars() < max_num_vars {
                            let max = 1 << mle.num_free_vars();
                            let multiple = (1 << max_num_vars) / max;
                            index / multiple
                        } else {
                            index
                        };
                        // Access the MLE at that index. Pad with zeros
                        mle.get(index).unwrap_or(F::ZERO)
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                // Combine them into the accumulator.
                //
                // Note that the accumulator stores g(0), g(1), ..., g(d - 1)
                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = sum.reduce(|| F::ZERO, |acc, partial| acc + partial);

        Ok(SumcheckEvals(vec![sum; degree]))
    }
}

/// Index mle indices for an array of mles.
pub fn index_mle_indices_gate<F: Field>(mles: &mut [impl Mle<F>], index: usize) {
    mles.iter_mut().for_each(|mle| {
        mle.index_mle_indices(index);
    })
}

/// Compute the fully bound identity gate function, which is f_1(g, u) where the
/// output gate label variables are fully bound to the claim on the layer and
/// the input gate label variables are fully bound to the round challenges from
/// sumcheck.
///
/// Similar to [fold_wiring_into_beta_mle_identity_gate], this function utilizes
/// an inspiration of Rothblum's memory-efficient sumcheck trick in order to
/// compute the relevant beta values over the sparse wiring function by just
/// multiplying by the relevant inverses.
///
/// The function is also usable if parallelism is turned on, by having the
/// folding accumulator keep track of the previous state.
pub fn compute_fully_bound_identity_gate_function<F: Field>(
    nondataparallel_round_challenges: &[F],
    nondataparallel_claim_challenges_vec: &[&[F]],
    wiring: &[(u32, u32)],
    random_coefficients: &[F],
) -> F {
    let (inverses_round_challenges, one_minus_elem_inverted_round_challenges): (Vec<F>, Vec<F>) =
        nondataparallel_round_challenges
            .iter()
            .map(|elem| {
                let inverse = elem.invert().unwrap();
                let one_minus_elem_inverse = (F::ONE - elem).invert().unwrap();
                (inverse, one_minus_elem_inverse)
            })
            .unzip();

    let (inverses_vec_claim_challenges, one_minus_elem_inverted_vec_claim_challenges) =
        compute_inverses_vec_and_one_minus_inverted_vec(nondataparallel_claim_challenges_vec);

    let prev_aux_and_gate_function = cfg_iter!(wiring).fold(
        #[cfg(feature = "parallel")]
        || {
            (
                ((None::<&u32>, None::<&u32>), (None::<Vec<F>>, None::<F>)),
                F::ZERO,
            )
        },
        #[cfg(not(feature = "parallel"))]
        (
            ((None::<&u32>, None::<&u32>), (None::<Vec<F>>, None::<F>)),
            F::ZERO,
        ),
        |(maybe_previous_aux, current_accumulation_of_gate_value),
         (next_nonzero_output_gate_label, next_nonzero_input_gate_label)| {
            // If we values for the previous auxiliary information, then we know
            // that we are past the first initialization of the iterator, and
            // use these previous values to accumulate.
            if let (
                (Some(current_nonzero_output_gate_label), Some(current_nonzero_input_gate_label)),
                (
                    Some(current_beta_values_claim_challenges),
                    Some(current_beta_value_round_challenges),
                ),
            ) = maybe_previous_aux
            {
                // Between the previous input label and the next one, compute the
                // flipped bits and their values in order to decide which inverses
                // to multiply by.
                let flipped_bit_idx_and_values_round =
                    compute_flipped_bit_idx_and_values_lexicographic(
                        *current_nonzero_input_gate_label,
                        *next_nonzero_input_gate_label,
                    );

                // We do the same thing for the previous output gate label and
                // the next one.
                let flipped_bit_idx_and_values_claim =
                    compute_flipped_bit_idx_and_values_lexicographic(
                        *current_nonzero_output_gate_label,
                        *next_nonzero_output_gate_label,
                    );

                // Compute the next beta value by multiplying the current ones
                // by the appropriate inverses and elements -- this is
                // specifically for the round challenges over the input labels
                // as indices. We know there is only one, so we directly index
                // into our function.
                let next_beta_value_round = compute_next_beta_value_from_current(
                    &current_beta_value_round_challenges,
                    &inverses_round_challenges,
                    &one_minus_elem_inverted_round_challenges,
                    nondataparallel_round_challenges,
                    &flipped_bit_idx_and_values_round,
                );

                // Compute the next beta values by multiplying the current ones
                // by the appropriate inverses and elements -- this is
                // specifically for the claim challenges over the output labels
                // as indices.
                let next_beta_values_claim = compute_next_beta_values_vec_from_current(
                    &current_beta_values_claim_challenges,
                    &inverses_vec_claim_challenges,
                    &one_minus_elem_inverted_vec_claim_challenges,
                    nondataparallel_claim_challenges_vec,
                    &flipped_bit_idx_and_values_claim,
                );

                let rlc_over_claim_beta_values = next_beta_values_claim
                    .iter()
                    .zip(random_coefficients)
                    .fold(F::ZERO, |acc, (elem, random_coeff)| {
                        acc + (*elem * random_coeff)
                    });

                // We accumulate by adding the previous value of \beta(g, z)
                // * \beta(u, x) to the current one.
                let accumulation_of_gate_value_vec = current_accumulation_of_gate_value
                    + next_beta_value_round * rlc_over_claim_beta_values;

                (
                    (
                        (
                            Some(next_nonzero_output_gate_label),
                            Some(next_nonzero_input_gate_label),
                        ),
                        (Some(next_beta_values_claim), Some(next_beta_value_round)),
                    ),
                    accumulation_of_gate_value_vec,
                )
            } else {
                // If the previous auxiliary information is None, we are on our
                // first iteration of the iterator and therefore simply
                // initialize the beta function over the point. We send this
                // information into the accumulator for future rounds.
                let next_beta_value_round = BetaValues::compute_beta_over_challenge_and_index(
                    nondataparallel_round_challenges,
                    *next_nonzero_input_gate_label as usize,
                );

                let next_beta_values_claim = nondataparallel_claim_challenges_vec
                    .iter()
                    .map(|claim_point| {
                        BetaValues::compute_beta_over_challenge_and_index(
                            claim_point,
                            *next_nonzero_output_gate_label as usize,
                        )
                    })
                    .collect_vec();

                let rlc_over_claim_beta_values = next_beta_values_claim
                    .iter()
                    .zip(random_coefficients)
                    .fold(F::ZERO, |acc, (elem, random_coeff)| {
                        acc + (*elem * random_coeff)
                    });

                (
                    (
                        (
                            Some(next_nonzero_output_gate_label),
                            Some(next_nonzero_input_gate_label),
                        ),
                        (Some(next_beta_values_claim), Some(next_beta_value_round)),
                    ),
                    next_beta_value_round * rlc_over_claim_beta_values,
                )
            }
        },
    );

    // For RLC Claim aggregation, we have multiple claims and therefore multiple beta tables.
    #[cfg(feature = "parallel")]
    {
        prev_aux_and_gate_function
            .map(|(_, gate_function)| gate_function)
            .sum::<F>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let (_, gate_function_over_claims) = prev_aux_and_gate_function;
        gate_function_over_claims
    }
}

/// Similar to [compute_fully_bound_identity_gate_function], this
/// function uses the "Rothblum" trick in order to evaluate the
/// fully bound gate function in a streaming fashion.
pub fn compute_fully_bound_binary_gate_function<F: Field>(
    nondataparallel_round_u_challenges: &[F],
    nondataparallel_round_v_challenges: &[F],
    nondataparallel_claim_challenges_vec: &[&[F]],
    wiring: &[(u32, u32, u32)],
    random_coefficients: &[F],
) -> F {
    let (inverses_round_u_challenges, one_minus_elem_inverted_round_u_challenges): (
        Vec<F>,
        Vec<F>,
    ) = nondataparallel_round_u_challenges
        .iter()
        .map(|elem| {
            let inverse = elem.invert().unwrap();
            let one_minus_elem_inverse = (F::ONE - elem).invert().unwrap();
            (inverse, one_minus_elem_inverse)
        })
        .unzip();

    let (inverses_round_v_challenges, one_minus_elem_inverted_round_v_challenges): (
        Vec<F>,
        Vec<F>,
    ) = nondataparallel_round_v_challenges
        .iter()
        .map(|elem| {
            let inverse = elem.invert().unwrap();
            let one_minus_elem_inverse = (F::ONE - elem).invert().unwrap();
            (inverse, one_minus_elem_inverse)
        })
        .unzip();

    let (inverses_vec_claim_challenges, one_minus_elem_inverted_vec_claim_challenges) =
        compute_inverses_vec_and_one_minus_inverted_vec(nondataparallel_claim_challenges_vec);

    let prev_aux_and_gate_function = cfg_iter!(wiring).fold(
        #[cfg(feature = "parallel")]
        || {
            (
                (None::<(&u32, &u32, &u32)>, None::<(Vec<F>, F, F)>),
                F::ZERO,
            )
        },
        #[cfg(not(feature = "parallel"))]
        (
            (None::<(&u32, &u32, &u32)>, None::<(Vec<F>, F, F)>),
            F::ZERO,
        ),
        |(maybe_previous_aux, current_accumulation_of_gate_value), (next_z, next_x, next_y)| {
            // If we values for the previous auxiliary information, then we know
            // that we are past the first initialization of the iterator, and
            // use these previous values to accumulate.
            if let (
                Some((current_z, current_x, current_y)),
                Some((current_beta_gz_vec, current_beta_u, current_beta_v)),
            ) = maybe_previous_aux
            {
                // Between the previous input label and the next one, compute the
                // flipped bits and their values in order to decide which inverses
                // to multiply by.
                let flipped_bit_idx_and_values_u =
                    compute_flipped_bit_idx_and_values_lexicographic(*current_x, *next_x);
                let flipped_bit_idx_and_values_v =
                    compute_flipped_bit_idx_and_values_lexicographic(*current_y, *next_y);

                // We do the same thing for the previous output gate label and
                // the next one.
                let flipped_bit_idx_and_values_z =
                    compute_flipped_bit_idx_and_values_lexicographic(*current_z, *next_z);

                // Compute the next beta value by multiplying the current ones
                // by the appropriate inverses and elements -- this is
                // specifically for the round challenges over the input labels
                // as indices. We know there is only one, so we directly index
                // into our function.
                let next_beta_value_u = compute_next_beta_value_from_current(
                    &current_beta_u,
                    &inverses_round_u_challenges,
                    &one_minus_elem_inverted_round_u_challenges,
                    nondataparallel_round_u_challenges,
                    &flipped_bit_idx_and_values_u,
                );
                let next_beta_value_v = compute_next_beta_value_from_current(
                    &current_beta_v,
                    &inverses_round_v_challenges,
                    &one_minus_elem_inverted_round_v_challenges,
                    nondataparallel_round_v_challenges,
                    &flipped_bit_idx_and_values_v,
                );

                // Compute the next beta values by multiplying the current ones
                // by the appropriate inverses and elements -- this is
                // specifically for the claim challenges over the output labels
                // as indices.
                let next_beta_values_gz = compute_next_beta_values_vec_from_current(
                    &current_beta_gz_vec,
                    &inverses_vec_claim_challenges,
                    &one_minus_elem_inverted_vec_claim_challenges,
                    nondataparallel_claim_challenges_vec,
                    &flipped_bit_idx_and_values_z,
                );

                let rlc_over_claim_beta_values = next_beta_values_gz
                    .iter()
                    .zip(random_coefficients)
                    .fold(F::ZERO, |acc, (elem, random_coeff)| {
                        acc + (*elem * random_coeff)
                    });

                // We accumulate by adding the previous value of \beta(g, z)
                // * \beta(u, x) to the current one.
                let accumulation_of_gate_value_vec = current_accumulation_of_gate_value
                    + next_beta_value_u * next_beta_value_v * rlc_over_claim_beta_values;

                (
                    (
                        Some((next_z, next_x, next_y)),
                        Some((next_beta_values_gz, next_beta_value_u, next_beta_value_v)),
                    ),
                    accumulation_of_gate_value_vec,
                )
            } else {
                // If the previous auxiliary information is None, we are on our
                // first iteration of the iterator and therefore simply
                // initialize the beta function over the point. We send this
                // information into the accumulator for future rounds.
                let next_beta_value_round_u = BetaValues::compute_beta_over_challenge_and_index(
                    nondataparallel_round_u_challenges,
                    *next_x as usize,
                );
                let next_beta_value_round_v = BetaValues::compute_beta_over_challenge_and_index(
                    nondataparallel_round_v_challenges,
                    *next_y as usize,
                );

                let next_beta_values_claim = nondataparallel_claim_challenges_vec
                    .iter()
                    .map(|claim_point| {
                        BetaValues::compute_beta_over_challenge_and_index(
                            claim_point,
                            *next_z as usize,
                        )
                    })
                    .collect_vec();

                let rlc_over_claim_beta_values = next_beta_values_claim
                    .iter()
                    .zip(random_coefficients)
                    .fold(F::ZERO, |acc, (elem, random_coeff)| {
                        acc + (*elem * random_coeff)
                    });

                (
                    (
                        Some((next_z, next_x, next_y)),
                        Some((
                            next_beta_values_claim,
                            next_beta_value_round_u,
                            next_beta_value_round_v,
                        )),
                    ),
                    next_beta_value_round_u * next_beta_value_round_v * rlc_over_claim_beta_values,
                )
            }
        },
    );

    // For RLC Claim aggregation, we have multiple claims and therefore multiple beta tables.
    #[cfg(feature = "parallel")]
    {
        prev_aux_and_gate_function
            .map(|(_, gate_function)| gate_function)
            .sum::<F>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let (_, gate_function_over_claims) = prev_aux_and_gate_function;
        gate_function_over_claims
    }
}

/// When `dataparallel_aux` is None, this function computes the coefficients of
/// the MLE representing f_1(g, x) where f_1 is the multilinear extension of our
/// gate function (which on the boolean hypercube, f_1(a, b) = 1 if there exists
/// a gate such that the label b is the input routing to the label a, and 0
/// otherwise). The resulting vector should have num_nondataparallel_copies
/// entries.
///
/// When `dataparallel_aux` is Some(..), this function computes the coefficients
/// of the MLE representing \sum_{wiring}{f_2(p_2, x) * f_1(g, x)} where p_2 is
/// the dataparallel index. The resulting vector should have
/// num_dataparallel_copies entries.
///
/// The folding occurs by first iterating through the nonzero gates, and
/// computing the value \beta(g, z) for each output label z and the challenge
/// point g. Instead of computing the \beta value from scratch, we take
/// inspiration from the Rothblum sumcheck trick to just multiply by the
/// inverses and points necessary to go from one beta value to the next. This
/// conserves memory by not having to store the entire table (which is done to
/// achieve linear computation) while continuing to achieve amortized linear
/// computation for each progressive beta value as opposed to the O(log(n))
/// method used to compute it from scratch.
///
/// We have multiple `claim_points` when using RLC claim agg, and in this case,
/// we compute the random linear combination of the `random_coefficients` and
/// the `g` variables.
pub fn fold_wiring_into_beta_mle_identity_gate<F: Field>(
    wiring: &[(u32, u32)],
    claim_points: &[&[F]],
    num_vars_folded_vec: usize,
    random_coefficients: &[F],
) -> Vec<F> {
    // Precompute all the inverses necessary for each of the claim points.
    let (inverses_vec, one_minus_inverses_vec) =
        compute_inverses_vec_and_one_minus_inverted_vec(claim_points);

    // Initialize the folded vector of coefficients, whose size is dependent
    // on whether we are in the dataparallel case.
    let mut folded_vec = vec![F::ZERO; 1 << num_vars_folded_vec];
    // We start at the first nonzero gate and first beta value for each claim
    // challenge.
    let (mut current_nonzero_output_gate_label, mut current_nonzero_input_gate_label) = wiring[0];
    let mut current_beta_values = claim_points
        .iter()
        .map(|claim_point| {
            BetaValues::compute_beta_over_challenge_and_index(
                claim_point,
                current_nonzero_output_gate_label as usize,
            )
        })
        .collect_vec();
    let first_nonzero_gate_beta_rlc = current_beta_values
        .iter()
        .zip(random_coefficients)
        .fold(F::ZERO, |acc, (elem, random_coeff)| {
            acc + (*elem * random_coeff)
        });

    // If it is dataparallel, we add the value of the source MLE over all of the
    // copies in order to compute the appropriate sum. Each of these are
    // multiplied by the same beta value.
    folded_vec[current_nonzero_input_gate_label as usize] = first_nonzero_gate_beta_rlc;

    wiring.iter().skip(1).for_each(
        |(next_nonzero_output_gate_label, next_nonzero_input_gate_label)| {
            // Between the previous output label and the next one, compute the
            // flipped bits and their values in order to decide which inverses
            // to multiply by.
            let flipped_bit_idx_and_values = compute_flipped_bit_idx_and_values_lexicographic(
                current_nonzero_output_gate_label,
                *next_nonzero_output_gate_label,
            );
            let next_beta_values = compute_next_beta_values_vec_from_current(
                &current_beta_values,
                &inverses_vec,
                &one_minus_inverses_vec,
                claim_points,
                &flipped_bit_idx_and_values,
            );

            let beta_values_rlc = next_beta_values
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (elem, random_coeff)| {
                    acc + (*elem * random_coeff)
                });

            // If dataparallel, add all of the values of the source_mle over
            // each of the copies.
            folded_vec[*next_nonzero_input_gate_label as usize] += beta_values_rlc;
            current_nonzero_input_gate_label = *next_nonzero_input_gate_label;
            current_nonzero_output_gate_label = *next_nonzero_output_gate_label;
            current_beta_values = next_beta_values;
        },
    );

    folded_vec
}

/// This function uses the "Rothblum"-inspired sumcheck trick
/// in order to evaluate the necessary MLEs as defined in
/// [Libra] for phase 1 of binary gate sumcheck messages.
pub fn fold_binary_gate_wiring_into_mles_phase_1<F: Field>(
    wiring: &[(u32, u32, u32)],
    claim_points: &[&[F]],
    f2_x_mle: &DenseMle<F>,
    f3_y_mle: &DenseMle<F>,
    random_coefficients: &[F],
    gate_operation: BinaryOperation,
) -> (Vec<F>, Vec<F>) {
    let num_vars = f2_x_mle.num_free_vars();
    let (inverses, one_minus_elem_inverted) =
        compute_inverses_vec_and_one_minus_inverted_vec(claim_points);
    let (_, a_hg_lhs, a_hg_rhs) = cfg_into_iter!(wiring).fold(
        #[cfg(feature = "parallel")]
        || {
            (
                None::<(Vec<F>, u32)>,
                vec![F::ZERO; 1 << num_vars],
                vec![F::ZERO; 1 << num_vars],
            )
        },
        #[cfg(not(feature = "parallel"))]
        (
            None::<(Vec<F>, u32)>,
            vec![F::ZERO; 1 << num_vars],
            vec![F::ZERO; 1 << num_vars],
        ),
        |(maybe_current_beta_aux, mut acc_of_a_hg_lhs, mut acc_of_a_hg_rhs),
         (next_z, next_x, next_y)| {
            let next_beta_vec = {
                if let Some((current_beta_values, current_z_idx)) = maybe_current_beta_aux {
                    let flipped_bits_and_idx =
                        compute_flipped_bit_idx_and_values_lexicographic(current_z_idx, *next_z);
                    compute_next_beta_values_vec_from_current(
                        &current_beta_values,
                        &inverses,
                        &one_minus_elem_inverted,
                        claim_points,
                        &flipped_bits_and_idx,
                    )
                } else {
                    claim_points
                        .iter()
                        .map(|claim_point| {
                            BetaValues::compute_beta_over_challenge_and_index(
                                claim_point,
                                *next_z as usize,
                            )
                        })
                        .collect_vec()
                }
            };
            let beta_vec_rlc = next_beta_vec
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (random_coeff, beta)| {
                    acc + *random_coeff * beta
                });
            let f_3_at_y = f3_y_mle.get(*next_y as usize).unwrap_or(F::ZERO);
            acc_of_a_hg_rhs[*next_x as usize] += beta_vec_rlc * f_3_at_y;
            if gate_operation == BinaryOperation::Add {
                acc_of_a_hg_lhs[*next_x as usize] += beta_vec_rlc;
            }
            (
                Some((next_beta_vec, *next_z)),
                acc_of_a_hg_lhs,
                acc_of_a_hg_rhs,
            )
        },
    );
    (a_hg_lhs, a_hg_rhs)
}

/// This function uses the "Rothblum"-inspired sumcheck trick
/// in order to evaluate the necessary MLEs as defined in
/// [Libra] for phase 2 of binary gate sumcheck messages.
pub fn fold_binary_gate_wiring_into_mles_phase_2<F: Field>(
    wiring: &[(u32, u32, u32)],
    f2_at_u: F,
    u_claim: &[F],
    g1_claim_points: &[&[F]],
    random_coefficients: &[F],
    num_vars: usize,
    gate_operation: BinaryOperation,
) -> (Vec<F>, Vec<F>) {
    let (inverses_g1, one_minus_elem_inverted_g1) =
        compute_inverses_vec_and_one_minus_inverted_vec(g1_claim_points);
    let (inverses_u, one_minus_elem_inverted_u): (Vec<F>, Vec<F>) = u_claim
        .iter()
        .map(|point| (point.invert().unwrap(), (F::ONE - point).invert().unwrap()))
        .unzip();
    let (_, a_f1_lhs, a_f1_rhs) = cfg_into_iter!(wiring).fold(
        #[cfg(feature = "parallel")]
        || {
            (
                (None::<(Vec<F>, u32)>, None::<(F, u32)>),
                vec![F::ZERO; 1 << num_vars],
                vec![F::ZERO; 1 << num_vars],
            )
        },
        #[cfg(not(feature = "parallel"))]
        (
            (None::<(Vec<F>, u32)>, None::<(F, u32)>),
            vec![F::ZERO; 1 << num_vars],
            vec![F::ZERO; 1 << num_vars],
        ),
        |(maybe_current_beta_aux, mut acc_of_a_f1_lhs, mut acc_of_a_f1_rhs),
         (next_z, next_x, next_y)| {
            let (next_beta_vec_g1, next_beta_u) = {
                if let (
                    Some((current_beta_values_g1, current_z_idx)),
                    Some((current_beta_value_u, current_x_idx)),
                ) = maybe_current_beta_aux
                {
                    let flipped_bits_and_idx_g1 =
                        compute_flipped_bit_idx_and_values_lexicographic(current_z_idx, *next_z);
                    let flipped_bits_and_idx_u =
                        compute_flipped_bit_idx_and_values_lexicographic(current_x_idx, *next_x);
                    let next_beta_vec_g1 = compute_next_beta_values_vec_from_current(
                        &current_beta_values_g1,
                        &inverses_g1,
                        &one_minus_elem_inverted_g1,
                        g1_claim_points,
                        &flipped_bits_and_idx_g1,
                    );
                    let next_beta_u = compute_next_beta_value_from_current(
                        &current_beta_value_u,
                        &inverses_u,
                        &one_minus_elem_inverted_u,
                        u_claim,
                        &flipped_bits_and_idx_u,
                    );
                    (next_beta_vec_g1, next_beta_u)
                } else {
                    let next_beta_vec_g1 = g1_claim_points
                        .iter()
                        .map(|claim_point| {
                            BetaValues::compute_beta_over_challenge_and_index(
                                claim_point,
                                *next_z as usize,
                            )
                        })
                        .collect_vec();
                    let next_beta_u = BetaValues::compute_beta_over_challenge_and_index(
                        u_claim,
                        *next_x as usize,
                    );
                    (next_beta_vec_g1, next_beta_u)
                }
            };
            let beta_vec_rlc_g1 = next_beta_vec_g1
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (random_coeff, beta)| {
                    acc + *random_coeff * beta
                });
            let adder = beta_vec_rlc_g1 * next_beta_u;
            acc_of_a_f1_lhs[*next_y as usize] += adder * f2_at_u;
            if gate_operation == BinaryOperation::Add {
                acc_of_a_f1_rhs[*next_y as usize] += adder;
            }
            (
                (
                    Some((next_beta_vec_g1, *next_z)),
                    Some((next_beta_u, *next_x)),
                ),
                acc_of_a_f1_lhs,
                acc_of_a_f1_rhs,
            )
        },
    );
    (a_f1_lhs, a_f1_rhs)
}

/// This function uses the "Rothblum"-inspired sumcheck trick
/// in order to evaluate the necessary MLEs as defined in
/// [Libra] for the dataparallel phase of an identity gate.
pub fn fold_wiring_into_dataparallel_beta_mle_identity_gate<F: Field>(
    wiring: &[(u32, u32)],
    claim_points: &[&[F]],
    num_dataparallel_vars: usize,
    source_mle: &DenseMle<F>,
    random_coefficients: &[F],
) -> (Vec<F>, Vec<F>) {
    //sum_{x, z}{\beta(g2g1, p2z) * f1(z, x, y) * Vi(p2, x)}
    // Precompute all the inverses necessary for each of the claim points.
    let (inverses_vec, one_minus_inverses_vec) =
        compute_inverses_vec_and_one_minus_inverted_vec(claim_points);

    // Initialize the folded vector of coefficients, whose size is dependent
    // on whether we are in the dataparallel case.
    let mut folded_vec_mle = vec![F::ZERO; 1 << num_dataparallel_vars];
    let mut folded_vec_beta = vec![F::ZERO; 1 << num_dataparallel_vars];

    // We start at the first nonzero gate and first beta value for each claim
    // challenge.
    let (current_nonzero_output_gate_label, current_nonzero_input_gate_label) = wiring[0];
    let mut current_idx_of_beta = current_nonzero_output_gate_label;
    let idx_of_mle = current_nonzero_input_gate_label;
    let mut current_beta_values = claim_points
        .iter()
        .map(|claim_point| {
            BetaValues::compute_beta_over_challenge_and_index(
                claim_point,
                current_nonzero_output_gate_label as usize,
            )
        })
        .collect_vec();
    let first_nonzero_gate_beta_rlc = current_beta_values
        .iter()
        .zip(random_coefficients)
        .fold(F::ZERO, |acc, (elem, random_coeff)| {
            acc + (*elem * random_coeff)
        });

    folded_vec_beta[0] = first_nonzero_gate_beta_rlc;
    folded_vec_mle[0] = source_mle.get(idx_of_mle as usize).unwrap();

    // We go through the first iteration of the dataparallel bits and compute
    // the appropriate beta values to add to the folded vector.
    let num_nondataparallel_coefficients_in_result =
        1 << (claim_points[0].len() - num_dataparallel_vars);
    let num_nondataparallel_coefficients_in_mle =
        1 << (source_mle.num_free_vars() - num_dataparallel_vars);

    (0..(1 << num_dataparallel_vars))
        .skip(1)
        .for_each(|dataparallel_copy_index| {
            dbg!(&current_nonzero_output_gate_label);
            let next_idx_of_beta = (dataparallel_copy_index
                * num_nondataparallel_coefficients_in_result)
                + current_nonzero_output_gate_label as usize;
            let idx_of_mle = (dataparallel_copy_index * num_nondataparallel_coefficients_in_mle)
                + current_nonzero_input_gate_label as usize;
            let source_value_at_idx = source_mle.get(idx_of_mle).unwrap();
            let flipped_bits_and_indices = compute_flipped_bit_idx_and_values_lexicographic(
                current_idx_of_beta,
                next_idx_of_beta as u32,
            );
            let next_beta_values = compute_next_beta_values_vec_from_current(
                &current_beta_values,
                &inverses_vec,
                &one_minus_inverses_vec,
                claim_points,
                &flipped_bits_and_indices,
            );
            let next_beta_rlc = next_beta_values
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (elem, random_coeff)| {
                    acc + (*elem * random_coeff)
                });
            folded_vec_beta[dataparallel_copy_index] += next_beta_rlc;
            folded_vec_mle[dataparallel_copy_index] += source_value_at_idx;
            current_idx_of_beta = next_idx_of_beta as u32;
            current_beta_values = next_beta_values;
        });

    (0..(1 << num_dataparallel_vars)).for_each(|dataparallel_copy_index| {
        wiring.iter().skip(1).for_each(
            |(next_nonzero_output_gate_label, next_nonzero_input_gate_label)| {
                let next_idx_of_beta = (dataparallel_copy_index
                    * num_nondataparallel_coefficients_in_result)
                    + *next_nonzero_output_gate_label as usize;
                let idx_of_mle = (dataparallel_copy_index
                    * num_nondataparallel_coefficients_in_mle)
                    + *next_nonzero_input_gate_label as usize;
                let source_value_at_idx = source_mle.get(idx_of_mle).unwrap();
                let flipped_bits_and_indices = compute_flipped_bit_idx_and_values_lexicographic(
                    current_idx_of_beta,
                    next_idx_of_beta as u32,
                );
                let next_beta_values = compute_next_beta_values_vec_from_current(
                    &current_beta_values,
                    &inverses_vec,
                    &one_minus_inverses_vec,
                    claim_points,
                    &flipped_bits_and_indices,
                );
                let next_beta_rlc = next_beta_values
                    .iter()
                    .zip(random_coefficients)
                    .fold(F::ZERO, |acc, (elem, random_coeff)| {
                        acc + (*elem * random_coeff)
                    });
                folded_vec_beta[dataparallel_copy_index] += next_beta_rlc;
                folded_vec_mle[dataparallel_copy_index] += source_value_at_idx;
                current_idx_of_beta = next_idx_of_beta as u32;
                current_beta_values = next_beta_values;
            },
        )
    });

    (folded_vec_beta, folded_vec_mle)
}

/// Compute sumcheck message without a beta table.
pub fn compute_sumcheck_message_no_beta_table<F: Field>(
    mles: &[&impl Mle<F>],
    round_index: usize,
    degree: usize,
) -> Result<Vec<F>, GateError> {
    // Go through all of the MLEs being multiplied together on the LHS and
    // see if any of them contain an IV
    let independent_variable = mles
        .iter()
        .map(|mle| mle.mle_indices().contains(&MleIndex::Indexed(round_index)))
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;

    let eval = evaluate_mle_product_no_beta_table(mles, independent_variable, degree).unwrap();

    let SumcheckEvals(evaluations) = eval;

    Ok(evaluations)
}

/// Get the evals for a binary gate specified by the BinaryOperation. Note that
/// this specifically refers to computing the prover message while binding the
/// dataparallel bits of a `IdentityGate` expression.
pub fn compute_sumcheck_message_data_parallel_identity_gate<F: Field>(
    source_mle: &DenseMle<F>,
    wiring: &[(u32, u32)],
    num_dataparallel_vars: usize,
    challenges_vec: &[&[F]],
    random_coefficients: &[F],
) -> Result<Vec<F>, GateError> {
    let (inverses_vec, one_minus_elem_inverted_vec) =
        compute_inverses_vec_and_one_minus_inverted_vec(challenges_vec);

    // UPDATE COMMMENT HERE
    let degree = 2;

    // There is an independent variable, and we must extract `degree`
    // evaluations of it, over `0..degree`.
    let eval_count = degree + 1;

    let num_dataparallel_copies_mid = 1 << (num_dataparallel_vars - 1);

    let num_nondataparallel_coeffs_source =
        1 << (source_mle.num_free_vars() - num_dataparallel_vars);
    let num_z_coeffs = 1 << (challenges_vec[0].len() - num_dataparallel_vars);
    let scaled_wirings = cfg_into_iter!((0..num_dataparallel_copies_mid))
        .flat_map(|p2_idx| {
            wiring
                .iter()
                .map(|(z, x)| {
                    (
                        num_z_coeffs * p2_idx + z,
                        num_nondataparallel_coeffs_source * p2_idx + x,
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let evals = cfg_into_iter!(scaled_wirings).fold(
        #[cfg(feature = "parallel")]
        || (vec![F::ZERO; eval_count], None::<(Vec<F>, u32)>),
        #[cfg(not(feature = "parallel"))]
        (vec![F::ZERO; eval_count], None::<(Vec<F>, u32)>),
        |(mut acc, maybe_current_beta_aux), (next_scaled_z, next_scaled_x)| {
            let next_beta_values_at_0 =
                if let Some((current_beta_values, current_scaled_z)) = maybe_current_beta_aux {
                    let flipped_bits_and_idx = compute_flipped_bit_idx_and_values_lexicographic(
                        current_scaled_z,
                        next_scaled_z,
                    );
                    compute_next_beta_values_vec_from_current(
                        &current_beta_values,
                        &inverses_vec,
                        &one_minus_elem_inverted_vec,
                        challenges_vec,
                        &flipped_bits_and_idx,
                    )
                } else {
                    challenges_vec
                        .iter()
                        .map(|challenge| {
                            BetaValues::compute_beta_over_challenge_and_index(
                                challenge,
                                next_scaled_z as usize,
                            )
                        })
                        .collect_vec()
                };
            let next_beta_values_at_1 = next_beta_values_at_0
                .iter()
                .zip(
                    challenges_vec
                        .iter()
                        .zip(one_minus_elem_inverted_vec.iter()),
                )
                .map(|(beta_at_0, (challenges, one_minus_inverses))| {
                    *beta_at_0 * one_minus_inverses[0] * challenges[0]
                })
                .collect_vec();
            dbg!(&next_beta_values_at_0, &next_beta_values_at_1);
            dbg!(
                &next_beta_values_at_0[0].neg(),
                &next_beta_values_at_1[0].neg()
            );

            let rlc_beta_values_at_0 = next_beta_values_at_0
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (beta_val, random_coeff)| {
                    acc + *beta_val * random_coeff
                });
            let rlc_beta_values_at_1 = next_beta_values_at_1
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (beta_val, random_coeff)| {
                    acc + *beta_val * random_coeff
                });
            let linear_diff_betas = rlc_beta_values_at_1 - rlc_beta_values_at_0;
            let beta_evals_p2_z =
                std::iter::successors(Some(rlc_beta_values_at_1), move |rlc_beta_values_prev| {
                    Some(*rlc_beta_values_prev + linear_diff_betas)
                });
            let all_beta_evals_p2_z = std::iter::once(rlc_beta_values_at_0).chain(beta_evals_p2_z);

            // Compute f_2((A, p_2), x) Note that the bookkeeping table
            // is big-endian, so we shift by idx * (number of non
            // dataparallel vars) to index into the correct copy.
            let source_0_p2_x = source_mle.get(next_scaled_x as usize).unwrap();
            let source_1_p2_x = if source_mle.num_free_vars() != 0 {
                source_mle
                    .get(
                        (next_scaled_x
                            + (num_nondataparallel_coeffs_source * num_dataparallel_copies_mid))
                            as usize,
                    )
                    .unwrap()
            } else {
                source_0_p2_x
            };
            let linear_diff_f2 = source_1_p2_x - source_0_p2_x;

            let f2_evals_p2_x = std::iter::successors(Some(source_1_p2_x), move |f2_prev_p2_x| {
                Some(*f2_prev_p2_x + linear_diff_f2)
            });
            let all_f2_evals_p2_x = std::iter::once(source_0_p2_x).chain(f2_evals_p2_x);

            // The evals we want are simply the element-wise product
            // of the accessed evals
            let g1_z_times_source_p2_x = all_beta_evals_p2_z
                .zip(all_f2_evals_p2_x)
                .map(|(g1_z_eval, source_eval)| g1_z_eval * source_eval);

            let evals_iter: Box<dyn Iterator<Item = F>> = Box::new(g1_z_times_source_p2_x);

            acc.iter_mut()
                .zip(evals_iter)
                .for_each(|(acc, eval)| *acc += eval);
            (acc, Some((next_beta_values_at_0, next_scaled_z)))
        },
    );

    #[cfg(feature = "parallel")]
    {
        let evals = evals.fold(
            || vec![F::ZERO; eval_count],
            |mut acc, (partial, _, _)| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );
        Ok(evals)
    }
    Ok(evals.0)
}

/// Get the evals for a binary gate specified by the BinaryOperation. Note that
/// this specifically refers to computing the prover message while binding the
/// dataparallel bits of a `Gate` expression.
pub fn compute_sumcheck_message_data_parallel_gate<F: Field>(
    f2_p2_x: &DenseMle<F>,
    f3_p2_y: &DenseMle<F>,
    operation: BinaryOperation,
    wiring: &[(u32, u32, u32)],
    num_dataparallel_vars: usize,
    challenges_vec: &[&[F]],
    random_coefficients: &[F],
) -> Result<Vec<F>, GateError> {
    let (inverses_vec, one_minus_elem_inverted_vec) =
        compute_inverses_vec_and_one_minus_inverted_vec(challenges_vec);

    // When we have an add gate, we can distribute the beta table over the
    // dataparallel challenges so we only multiply to the function with the x
    // variables or y variables one at a time.
    //
    // When we have a mul gate, we have to multiply the beta table over the
    // dataparallel challenges with the function on the x variables and the
    // function on the y variables.
    let degree = match operation {
        BinaryOperation::Add => 2,
        BinaryOperation::Mul => 3,
    };

    // There is an independent variable, and we must extract `degree`
    // evaluations of it, over `0..degree`.
    let eval_count = degree + 1;

    let num_dataparallel_copies_mid = 1 << (num_dataparallel_vars - 1);

    let num_nondataparallel_coeffs_f2_x = 1 << (f2_p2_x.num_free_vars() - num_dataparallel_vars);
    let num_nondataparallel_coeffs_f3_y = 1 << (f3_p2_y.num_free_vars() - num_dataparallel_vars);
    let num_z_coeffs = 1 << (challenges_vec[0].len() - num_dataparallel_vars);
    let scaled_wirings = cfg_into_iter!((0..num_dataparallel_copies_mid))
        .flat_map(|p2_idx| {
            wiring
                .iter()
                .map(|(z, x, y)| {
                    (
                        num_z_coeffs * p2_idx + z,
                        num_nondataparallel_coeffs_f2_x * p2_idx + x,
                        num_nondataparallel_coeffs_f3_y * p2_idx + y,
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    let evals = cfg_into_iter!(scaled_wirings).fold(
        #[cfg(feature = "parallel")]
        || (vec![F::ZERO; eval_count], None::<(Vec<F>, u32)>),
        #[cfg(not(feature = "parallel"))]
        (vec![F::ZERO; eval_count], None::<(Vec<F>, u32)>),
        |(mut acc, maybe_current_beta_aux), (scaled_z, scaled_x, scaled_y)| {
            let next_beta_values_at_0 = if let Some((current_beta_values, current_scaled_z)) =
                maybe_current_beta_aux
            {
                let flipped_bits_and_idx =
                    compute_flipped_bit_idx_and_values_lexicographic(current_scaled_z, scaled_z);
                compute_next_beta_values_vec_from_current(
                    &current_beta_values,
                    &inverses_vec,
                    &one_minus_elem_inverted_vec,
                    challenges_vec,
                    &flipped_bits_and_idx,
                )
            } else {
                challenges_vec
                    .iter()
                    .map(|challenge| {
                        BetaValues::compute_beta_over_challenge_and_index(
                            challenge,
                            scaled_z as usize,
                        )
                    })
                    .collect_vec()
            };
            let next_beta_values_at_1 = next_beta_values_at_0
                .iter()
                .zip(
                    challenges_vec
                        .iter()
                        .zip(one_minus_elem_inverted_vec.iter()),
                )
                .map(|(beta_at_0, (challenges, one_minus_inverses))| {
                    *beta_at_0 * one_minus_inverses[0] * challenges[0]
                })
                .collect_vec();

            let rlc_beta_values_at_0 = next_beta_values_at_0
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (beta_val, random_coeff)| {
                    acc + *beta_val * random_coeff
                });
            let rlc_beta_values_at_1 = next_beta_values_at_1
                .iter()
                .zip(random_coefficients)
                .fold(F::ZERO, |acc, (beta_val, random_coeff)| {
                    acc + *beta_val * random_coeff
                });
            let linear_diff_betas = rlc_beta_values_at_1 - rlc_beta_values_at_0;
            let beta_evals_p2_z =
                std::iter::successors(Some(rlc_beta_values_at_1), move |rlc_beta_values_prev| {
                    Some(*rlc_beta_values_prev + linear_diff_betas)
                });
            let all_beta_evals_p2_z = std::iter::once(rlc_beta_values_at_0).chain(beta_evals_p2_z);

            // Compute f_2((A, p_2), x) Note that the bookkeeping table
            // is big-endian, so we shift by idx * (number of non
            // dataparallel vars) to index into the correct copy.
            let f2_0_p2_x = f2_p2_x.get(scaled_x as usize).unwrap();
            let f2_1_p2_x = if f2_p2_x.num_free_vars() != 0 {
                f2_p2_x
                    .get(
                        (scaled_x + (num_nondataparallel_coeffs_f2_x * num_dataparallel_copies_mid))
                            as usize,
                    )
                    .unwrap()
            } else {
                f2_0_p2_x
            };
            let linear_diff_f2 = f2_1_p2_x - f2_0_p2_x;

            let f2_evals_p2_x = std::iter::successors(Some(f2_1_p2_x), move |f2_prev_p2_x| {
                Some(*f2_prev_p2_x + linear_diff_f2)
            });
            let all_f2_evals_p2_x = std::iter::once(f2_0_p2_x).chain(f2_evals_p2_x);

            // Compute f_3((A, p_2), y). Note that the bookkeeping table is
            // big-endian, so we shift by `idx * (number of non dataparallel
            // vars)` to index into the correct copy.
            let f3_0_p2_y = f3_p2_y.get(scaled_y as usize).unwrap();
            let f3_1_p2_y = if f3_p2_y.num_free_vars() != 0 {
                f3_p2_y
                    .get(
                        (scaled_y + (num_nondataparallel_coeffs_f3_y * num_dataparallel_copies_mid))
                            as usize,
                    )
                    .unwrap()
            } else {
                f3_0_p2_y
            };
            let linear_diff_f3 = f3_1_p2_y - f3_0_p2_y;

            let f3_evals_p2_y = std::iter::successors(Some(f3_1_p2_y), move |f3_prev_p2_y| {
                Some(*f3_prev_p2_y + linear_diff_f3)
            });
            let all_f3_evals_p2_y = std::iter::once(f3_0_p2_y).chain(f3_evals_p2_y);

            // The evals we want are simply the element-wise product
            // of the accessed evals
            let g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y = all_beta_evals_p2_z
                .zip(all_f2_evals_p2_x.zip(all_f3_evals_p2_y))
                .map(|(g1_z_eval, (f2_eval, f3_eval))| {
                    g1_z_eval * operation.perform_operation(f2_eval, f3_eval)
                });

            let evals_iter: Box<dyn Iterator<Item = F>> =
                Box::new(g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y);

            acc.iter_mut()
                .zip(evals_iter)
                .for_each(|(acc, eval)| *acc += eval);
            (acc, Some((next_beta_values_at_0, scaled_z)))
        },
    );

    #[cfg(feature = "parallel")]
    {
        let evals = evals.fold(
            || vec![F::ZERO; eval_count],
            |mut acc, (partial, _, _)| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );
        Ok(evals)
    }
    Ok(evals.0)
}

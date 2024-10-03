use ark_std::cfg_into_iter;
use itertools::Itertools;

use std::{cmp::max, fmt::Debug};

use crate::{
    mle::{betavalues::BetaValues, Mle},
    sumcheck::*,
};
use remainder_shared_types::Field;

use crate::mle::{dense::DenseMle, MleIndex};
use thiserror::Error;

use super::BinaryOperation;
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Error handling for gate mle construction.
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("phase 1 not initialized")]
    /// Error when initializing the first phase, which is when we bind the "x" bits.
    Phase1InitError,
    #[error("phase 2 not initialized")]
    /// Error when initializing the second phase, which is when we bind the "y" bits.
    Phase2InitError,
    #[error("mle not fully bound")]
    /// We are on the last round of sumcheck and want to grab claims, but the MLE is not fully bound
    /// which should not be the case for the last round of sumcheck.
    MleNotFullyBoundError,
    #[error("empty list for lhs or rhs")]
    /// We are initializing a gate on something that does not have either a left or right side of the expression.
    EmptyMleList,
    #[error("bound indices fail to match challenge")]
    /// When checking the last round of sumcheck, the challenges don't match what is bound to the MLE.
    EvaluateBoundIndicesDontMatch,
    #[error("beta table associated is not indexed")]
    /// The beta table we are working with doesn't have numbered indices but we need labeled bits!
    #[allow(dead_code)]
    BetaTableNotIndexed,
}

/// Given (possibly half-fixed) bookkeeping tables of the MLEs which are multiplied,
/// e.g. V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// computes g_k(x) = \sum_{b_{k + 1}, ..., b_n} V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// at `degree + 1` points.
///
/// ## Arguments
/// * `mle_refs` - MLEs pointing to the actual bookkeeping tables for the above
/// * `independent_variable` - whether the `x` from above resides within at least one of the `mle_refs`
/// * `degree` - degree of `g_k(x)`, i.e. number of evaluations to send (minus one!)
pub fn evaluate_mle_ref_product_no_beta_table<F: Field>(
    mle_refs: &[&impl Mle<F>],
    independent_variable: bool,
    degree: usize,
) -> Result<SumcheckEvals<F>, MleError> {
    // --- Gets the total number of free variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_free_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    if independent_variable {
        // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
        let eval_count = degree + 1;

        // iterate across all pairs of evaluations
        let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
            #[cfg(feature = "parallel")]
            || vec![F::ZERO; eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::ZERO; eval_count],
            |mut acc, index| {
                //get the product of all evaluations over 0/1/..degree
                let evals = mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let zero = F::ZERO;
                        let index = if mle_ref.num_free_vars() < max_num_vars {
                            let max = 1 << mle_ref.num_free_vars();
                            (index * 2) % max
                        } else {
                            index * 2
                        };
                        let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);
                        let second = if mle_ref.num_free_vars() != 0 {
                            *mle_ref.bookkeeping_table().get(index + 1).unwrap_or(&zero)
                        } else {
                            first
                        };

                        let step = second - first;

                        let successors =
                            std::iter::successors(Some(second), move |item| Some(*item + step));
                        // iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
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
                let product = mle_refs
                    .iter()
                    // Result of this `map()`: A list of evaluations of the MLEs at `index`
                    .map(|mle_ref| {
                        let index = if mle_ref.num_free_vars() < max_num_vars {
                            let max = 1 << mle_ref.num_free_vars();
                            index % max
                        } else {
                            index
                        };
                        // --- Access the MLE at that index. Pad with zeros ---
                        mle_ref
                            .bookkeeping_table()
                            .get(index)
                            .cloned()
                            .unwrap_or(F::ZERO)
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                // --- Combine them into the accumulator ---
                // Note that the accumulator stores g(0), g(1), ..., g(d - 1)
                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = sum.reduce(|| F::ZERO, |acc, partial| acc + partial);

        Ok(SumcheckEvals(vec![sum; degree]))
    }
}

/// checks whether mle was bound correctly to all the challenge points!!!!!!!!!!
pub fn check_fully_bound<F: Field>(
    mle_refs: &mut [impl Mle<F>],
    challenges: Vec<F>,
) -> Result<F, GateError> {
    let mles_bound: Vec<bool> = mle_refs
        .iter()
        .map(|mle_ref| {
            let indices = mle_ref
                .mle_indices()
                .iter()
                .filter_map(|index| match index {
                    MleIndex::Bound(chal, index) => Some((*chal, index)),
                    _ => None,
                })
                .collect_vec();

            let (indices, _): (Vec<_>, Vec<usize>) = indices.into_iter().unzip();

            indices == challenges
        })
        .collect();

    if mles_bound.contains(&false) {
        return Err(GateError::EvaluateBoundIndicesDontMatch);
    }

    mle_refs.iter_mut().try_fold(F::ONE, |acc, mle_ref| {
        // --- Accumulate either errors or multiply ---
        if mle_ref.bookkeeping_table().len() != 1 {
            return Err(GateError::MleNotFullyBoundError);
        }
        Ok(acc * mle_ref.bookkeeping_table()[0])
    })
}

/// Index mle indices for an array of mles.
pub fn index_mle_indices_gate<F: Field>(mle_refs: &mut [impl Mle<F>], index: usize) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        mle_ref.index_mle_indices(index);
    })
}

/// Fixes variable for the MLEs of a round of sumcheck for add/mul gates.
pub fn bind_round_gate<F: Field>(
    round_index: usize,
    challenge: F,
    mle_refs: &mut [Vec<DenseMle<F>>],
) {
    mle_refs.iter_mut().for_each(|mle_ref_vec| {
        mle_ref_vec.iter_mut().for_each(|mle_ref| {
            mle_ref.fix_variable(round_index - 1, challenge);
        })
    });
}

/// Computes a round of the sumcheck protocol on a binary gate layer.
pub fn compute_sumcheck_message_gate<F: Field>(
    round_index: usize,
    mle_refs: &[Vec<&DenseMle<F>>],
) -> Vec<F> {
    let max_deg = mle_refs.iter().fold(0, |acc, elem| max(acc, elem.len()));
    let evals_vec = mle_refs
        .iter()
        .map(|mle_vec| {
            compute_sumcheck_message_no_beta_table(mle_vec, round_index, max_deg).unwrap()
        })
        .collect_vec();

    let final_evals = evals_vec
        .clone()
        .into_iter()
        .skip(1)
        .fold(SumcheckEvals(evals_vec[0].clone()), |acc, elem| {
            acc + SumcheckEvals(elem)
        });
    let SumcheckEvals(final_vec_evals) = final_evals;
    final_vec_evals
}

/// Fixes variable for the MLEs of a round of sumcheck for identity gates.
pub fn bind_round_identity<F: Field>(
    round_index: usize,
    challenge: F,
    mle_refs: &mut [DenseMle<F>],
) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        mle_ref.fix_variable(round_index - 1, challenge);
    });
}

/// Computes a round of sumcheck protocol on a unary gate layer.
pub fn compute_sumcheck_message_identity<F: Field>(
    round_index: usize,
    mle_refs: &[&DenseMle<F>],
) -> Result<Vec<F>, GateError> {
    let independent_variable = mle_refs
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::Indexed(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let evals =
        evaluate_mle_ref_product_no_beta_table(mle_refs, independent_variable, mle_refs.len())
            .unwrap();
    let SumcheckEvals(evaluations) = evals;
    Ok(evaluations)
}

/// Fully evaluates a gate expression (for both the batched and non-batched case, add and mul gates).
pub fn compute_full_gate<F: Field>(
    challenges: Vec<F>,
    lhs: &mut DenseMle<F>,
    rhs: &mut DenseMle<F>,
    nonzero_gates: &[(usize, usize, usize)],
    copy_bits: usize,
) -> F {
    // Split the challenges into which ones are for batched bits, which ones are for others.
    let mut copy_chals: Vec<F> = vec![];
    let mut z_chals: Vec<F> = vec![];
    challenges.into_iter().enumerate().for_each(|(idx, chal)| {
        if (0..copy_bits).contains(&idx) {
            copy_chals.push(chal);
        } else {
            z_chals.push(chal);
        }
    });

    // If the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on z.
    let beta_g = BetaValues::new_beta_equality_mle(z_chals);
    let zero = F::ZERO;

    // Literally summing over everything else (x, y).
    if copy_bits == 0 {
        nonzero_gates
            .iter()
            .copied()
            .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                let ux = lhs.bookkeeping_table().get(x_ind).unwrap_or(&zero);
                let vy = rhs.bookkeeping_table().get(y_ind).unwrap_or(&zero);
                acc + gz * (*ux + *vy)
            })
    } else {
        let num_copy_idx = 1 << copy_bits;
        // If the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on p2.
        let beta_g2 = BetaValues::new_beta_equality_mle(copy_chals);
        {
            // Sum over everything else, outer sum being over p2, inner sum over (x, y).
            (0..(1 << num_copy_idx)).fold(F::ZERO, |acc_outer, idx| {
                let g2 = *beta_g2.bookkeeping_table().get(idx).unwrap_or(&F::ZERO);
                let inner_sum =
                    nonzero_gates
                        .iter()
                        .copied()
                        .fold(F::ZERO, |acc, (z_ind, x_ind, y_ind)| {
                            let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                            let ux = lhs
                                .bookkeeping_table()
                                .get(idx + (x_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            let vy = rhs
                                .bookkeeping_table()
                                .get(idx + (y_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            acc + gz * (*ux + *vy)
                        });
                acc_outer + (g2 * inner_sum)
            })
        }
    }
}

/// Compute the full value of the gate wiring function for an identity gate.
pub fn compute_full_gate_identity<F: Field>(
    challenges: Vec<F>,
    mle_ref: &mut DenseMle<F>,
    nonzero_gates: &[(usize, usize)],
    num_dataparallel_vars: usize,
) -> F {
    // Split the challenges into which ones are for batched bits, which ones are for others.
    let mut copy_chals: Vec<F> = vec![];
    let mut z_chals: Vec<F> = vec![];
    challenges.into_iter().enumerate().for_each(|(idx, chal)| {
        if (0..num_dataparallel_vars).contains(&idx) {
            copy_chals.push(chal);
        } else {
            z_chals.push(chal);
        }
    });

    // if the gate looks like f1(z, x)(f2(p2, x)) then this is the beta table for the challenges on z
    let beta_g = BetaValues::new_beta_equality_mle(z_chals);
    let zero = F::ZERO;

    if num_dataparallel_vars == 0 {
        nonzero_gates.iter().fold(F::ZERO, |acc, (z_ind, x_ind)| {
            let gz = *beta_g.bookkeeping_table().get(*z_ind).unwrap_or(&F::ZERO);
            let ux = mle_ref.bookkeeping_table().get(*x_ind).unwrap_or(&zero);
            acc + gz * (*ux)
        })
    } else {
        let num_copy_idx = 1 << num_dataparallel_vars;
        // If the gate looks like f1(z, x)f2(p2, x) then this is the beta table for the challenges on p2.
        let beta_g2 = BetaValues::new_beta_equality_mle(copy_chals);
        {
            // Sum over everything else, outer sum being over p2, inner sum over x.
            (0..(1 << num_copy_idx)).fold(F::ZERO, |acc_outer, idx| {
                let g2 = *beta_g2.bookkeeping_table().get(idx).unwrap_or(&F::ZERO);
                let inner_sum =
                    nonzero_gates
                        .iter()
                        .copied()
                        .fold(F::ZERO, |acc, (z_ind, x_ind)| {
                            let gz = *beta_g.bookkeeping_table().get(z_ind).unwrap_or(&F::ZERO);
                            let ux = mle_ref
                                .bookkeeping_table()
                                .get(idx + (x_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            acc + gz * (*ux)
                        });
                acc_outer + (g2 * inner_sum)
            })
        }
    }
}

/// Compute sumcheck message without a beta table.
pub fn compute_sumcheck_message_no_beta_table<F: Field>(
    mles: &[&impl Mle<F>],
    round_index: usize,
    degree: usize,
) -> Result<Vec<F>, GateError> {
    // --- Go through all of the MLEs being multiplied together on the LHS and see if any of them contain an IV ---
    let independent_variable = mles
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::Indexed(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let eval = evaluate_mle_ref_product_no_beta_table(mles, independent_variable, degree).unwrap();

    let SumcheckEvals(evaluations) = eval;

    Ok(evaluations)
}

/// Does all the necessary updates when proving a round for data parallel gate mles.
#[allow(clippy::too_many_arguments)]
pub fn prove_round_dataparallel_phase<F: Field>(
    lhs: &mut DenseMle<F>,
    rhs: &mut DenseMle<F>,
    beta_g1: &DenseMle<F>,
    beta_g2: &mut DenseMle<F>,
    round_index: usize,
    challenge: F,
    nonzero_gates: &[(usize, usize, usize)],
    num_dataparallel_bits: usize,
    operation: BinaryOperation,
) -> Result<Vec<F>, GateError> {
    beta_g2.fix_variable(round_index - 1, challenge);
    // Need to separately update these because the phase_lhs and phase_rhs has no version of them.
    lhs.fix_variable(round_index - 1, challenge);
    rhs.fix_variable(round_index - 1, challenge);
    compute_sumcheck_messages_data_parallel_gate(
        lhs,
        rhs,
        beta_g2,
        beta_g1,
        operation,
        nonzero_gates,
        num_dataparallel_bits,
    )
}

/// Get the evals for a binary gate specified by the BinaryOperation. Note that this specifically
/// refers to computing the prover message while binding the dataparallel bits of a `Gate`
/// expression.
pub fn compute_sumcheck_messages_data_parallel_gate<F: Field>(
    f2_p2_x: &DenseMle<F>,
    f3_p2_y: &DenseMle<F>,
    beta_g2: &DenseMle<F>,
    beta_g1: &DenseMle<F>,
    operation: BinaryOperation,
    nonzero_gates: &[(usize, usize, usize)],
    num_dataparallel_bits: usize,
) -> Result<Vec<F>, GateError> {
    // When we have an add gate, we can distribute the beta table over the dataparallel challenges
    // so we only multiply to the function with the x variables or y variables one at a time.
    // When we have a mul gate, we have to multiply the beta table over the dataparallel challenges
    // with the function on the x variables and the function on the y variables.
    let degree = match operation {
        BinaryOperation::Add => 2,
        BinaryOperation::Mul => 3,
    };

    // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`.
    let eval_count = degree + 1;

    // Iterate across all pairs of evaluations.
    let evals = cfg_into_iter!((0..1 << (num_dataparallel_bits - 1))).fold(
        #[cfg(feature = "parallel")]
        || vec![F::ZERO; eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::ZERO; eval_count],
        |mut acc, p2_idx| {
            // Compute the beta successors the same way it's done for each mle. Do it outside the loop
            // because it only needs to be done once per product of mles.
            let first = *beta_g2.bookkeeping_table().get(p2_idx * 2).unwrap();
            let second = if beta_g2.num_free_vars() != 0 {
                *beta_g2.bookkeeping_table().get(p2_idx * 2 + 1).unwrap()
            } else {
                first
            };
            let step = second - first;

            let beta_successors_snd =
                std::iter::successors(Some(second), move |item| Some(*item + step));
            // Iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1.
            let beta_successors = std::iter::once(first).chain(beta_successors_snd);
            let beta_iter: Box<dyn Iterator<Item = F>> = Box::new(beta_successors);

            let num_dataparallel_entries = 1 << num_dataparallel_bits;
            let inner_sum_successors = nonzero_gates
                .iter()
                .copied()
                .map(|(z, x, y)| {
                    let g1_z = beta_g1.mle[z];
                    let g1_z_successors = std::iter::successors(Some(g1_z), move |_| Some(g1_z));

                    // --- Compute f_2((A, p_2), x) ---
                    // --- Note that the bookkeeping table is little-endian, so we shift by `x * num_dataparallel_entries` ---
                    let f2_0_p2_x = *f2_p2_x
                        .bookkeeping_table()
                        .get((p2_idx * 2) + x * num_dataparallel_entries)
                        .unwrap();
                    let f2_1_p2_x = if f2_p2_x.num_free_vars() != 0 {
                        *f2_p2_x
                            .bookkeeping_table()
                            .get((p2_idx * 2 + 1) + x * num_dataparallel_entries)
                            .unwrap()
                    } else {
                        f2_0_p2_x
                    };
                    let linear_diff_f2 = f2_1_p2_x - f2_0_p2_x;

                    let f2_evals_p2_x =
                        std::iter::successors(Some(f2_1_p2_x), move |f2_prev_p2_x| {
                            Some(*f2_prev_p2_x + linear_diff_f2)
                        });
                    let all_f2_evals_p2_x = std::iter::once(f2_0_p2_x).chain(f2_evals_p2_x);

                    // --- Compute f_3((A, p_2), y) ---
                    // --- Note that the bookkeeping table is little-endian, so we shift by `y * num_dataparallel_entries` ---
                    let f3_0_p2_y = *f3_p2_y
                        .bookkeeping_table()
                        .get((p2_idx * 2) + y * num_dataparallel_entries)
                        .unwrap();
                    let f3_1_p2_y = if f3_p2_y.num_free_vars() != 0 {
                        *f3_p2_y
                            .bookkeeping_table()
                            .get((p2_idx * 2 + 1) + y * num_dataparallel_entries)
                            .unwrap()
                    } else {
                        f3_0_p2_y
                    };
                    let linear_diff_f3 = f3_1_p2_y - f3_0_p2_y;

                    let f3_evals_p2_y =
                        std::iter::successors(Some(f3_1_p2_y), move |f3_prev_p2_y| {
                            Some(*f3_prev_p2_y + linear_diff_f3)
                        });
                    let all_f3_evals_p2_y = std::iter::once(f3_0_p2_y).chain(f3_evals_p2_y);

                    // --- The evals we want are simply the element-wise product of the accessed evals ---
                    let g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y = g1_z_successors
                        .zip(all_f2_evals_p2_x.zip(all_f3_evals_p2_y))
                        .map(|(g1_z_eval, (f2_eval, f3_eval))| {
                            g1_z_eval * operation.perform_operation(f2_eval, f3_eval)
                        });

                    let evals_iter: Box<dyn Iterator<Item = F>> =
                        Box::new(g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y);

                    evals_iter
                })
                .reduce(|acc, successor| {
                    let add_successors = acc
                        .zip(successor)
                        .map(|(acc_eval, successor_eval)| acc_eval + successor_eval);

                    let add_iter: Box<dyn Iterator<Item = F>> = Box::new(add_successors);
                    add_iter
                })
                .unwrap();

            let evals = std::iter::once(inner_sum_successors)
                // chain the beta successors
                .chain(std::iter::once(beta_iter))
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
    Ok(evals)
}

/// Does all the necessary updates when proving a round for batched gate mles.
#[allow(clippy::too_many_arguments)]
pub fn prove_round_identity_gate_dataparallel_phase<F: Field>(
    src_mle: &mut DenseMle<F>,
    beta_g1: &DenseMle<F>,
    beta_g2: &mut DenseMle<F>,
    round_index: usize,
    challenge: F,
    nonzero_gates: &[(usize, usize)],
    num_dataparallel_bits: usize,
) -> Result<Vec<F>, GateError> {
    beta_g2.fix_variable(round_index - 1, challenge);
    // Need to separately update these because the phase_lhs and phase_rhs has no version of them.
    src_mle.fix_variable(round_index - 1, challenge);
    compute_sumcheck_messages_data_parallel_identity_gate(
        src_mle,
        beta_g2,
        beta_g1,
        nonzero_gates,
        num_dataparallel_bits,
    )
}

/// Get the evals for an identity gate. Note that this specifically
/// refers to computing the prover message while binding the dataparallel bits of a `Gate`
/// expression.
pub fn compute_sumcheck_messages_data_parallel_identity_gate<F: Field>(
    f2_p2_x: &DenseMle<F>,
    beta_g2: &DenseMle<F>,
    beta_g1: &DenseMle<F>,
    nonzero_gates: &[(usize, usize)],
    num_dataparallel_bits: usize,
) -> Result<Vec<F>, GateError> {
    // When we have an identity gate, we have to multiply the beta table over the dataparallel challenges
    // with the function on the x variables.
    let degree = 2;

    // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`.
    let eval_count = degree + 1;

    // Iterate across all pairs of evaluations.
    let evals = cfg_into_iter!((0..1 << (num_dataparallel_bits - 1))).fold(
        #[cfg(feature = "parallel")]
        || vec![F::ZERO; eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::ZERO; eval_count],
        |mut acc, p2_idx| {
            // Compute the beta successors the same way it's done for each mle. Do it outside the loop
            // because it only needs to be done once per product of mles.
            let first = *beta_g2.bookkeeping_table().get(p2_idx * 2).unwrap();
            let second = if beta_g2.num_free_vars() != 0 {
                *beta_g2.bookkeeping_table().get(p2_idx * 2 + 1).unwrap()
            } else {
                first
            };
            let step = second - first;

            let beta_successors_snd =
                std::iter::successors(Some(second), move |item| Some(*item + step));
            // Iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1.
            let beta_successors = std::iter::once(first).chain(beta_successors_snd);
            let beta_iter: Box<dyn Iterator<Item = F>> = Box::new(beta_successors);

            let num_dataparallel_entries = 1 << num_dataparallel_bits;
            let inner_sum_successors = nonzero_gates
                .iter()
                .copied()
                .map(|(z, x)| {
                    let g1_z = beta_g1.mle[z];
                    let g1_z_successors = std::iter::successors(Some(g1_z), move |_| Some(g1_z));

                    // --- Compute f_2((A, p_2), x) ---
                    // --- Note that the bookkeeping table is little-endian, so we shift by `x * num_dataparallel_entries` ---
                    let f2_0_p2_x = *f2_p2_x
                        .bookkeeping_table()
                        .get((p2_idx * 2) + x * num_dataparallel_entries)
                        .unwrap();
                    let f2_1_p2_x = if f2_p2_x.num_free_vars() != 0 {
                        *f2_p2_x
                            .bookkeeping_table()
                            .get((p2_idx * 2 + 1) + x * num_dataparallel_entries)
                            .unwrap()
                    } else {
                        f2_0_p2_x
                    };
                    let linear_diff_f2 = f2_1_p2_x - f2_0_p2_x;

                    let f2_evals_p2_x =
                        std::iter::successors(Some(f2_1_p2_x), move |f2_prev_p2_x| {
                            Some(*f2_prev_p2_x + linear_diff_f2)
                        });
                    let all_f2_evals_p2_x = std::iter::once(f2_0_p2_x).chain(f2_evals_p2_x);

                    // --- The evals we want are simply the element-wise product of the accessed evals ---
                    let g1_z_times_f2_evals_p2_x = g1_z_successors
                        .zip(all_f2_evals_p2_x)
                        .map(|(g1_z_eval, f2_eval)| g1_z_eval * f2_eval);

                    let evals_iter: Box<dyn Iterator<Item = F>> =
                        Box::new(g1_z_times_f2_evals_p2_x);

                    evals_iter
                })
                .reduce(|acc, successor| {
                    let add_successors = acc
                        .zip(successor)
                        .map(|(acc_eval, successor_eval)| acc_eval + successor_eval);

                    let add_iter: Box<dyn Iterator<Item = F>> = Box::new(add_successors);
                    add_iter
                })
                .unwrap();

            let evals = std::iter::once(inner_sum_successors)
                // chain the beta successors
                .chain(std::iter::once(beta_iter))
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
    Ok(evals)
}

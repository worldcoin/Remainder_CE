use ark_std::{cfg_into_iter, rand::Rng};
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    expression::ExpressionStandard,
    layer::{
        claims::ClaimError, layer_enum::LayerEnum, Claim, Layer, LayerBuilder, LayerError, LayerId,
        VerificationError,
    },
    mle::beta::BetaTable,
    prover::SumcheckProof,
    sumcheck::*,
};
use remainder_shared_types::{transcript::Transcript, FieldExt};

use super::{
    beta::compute_beta_over_two_challenges,
    dense::{DenseMle, DenseMleRef},
    MleIndex, MleRef,
};
use thiserror::Error;

/// Error handling for gate mle construction
#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("phase 1 not initialized")]
    Phase1InitError,
    #[error("phase 2 not initialized")]
    Phase2InitError,
    #[error("copy phase init error")]
    CopyPhaseInitError,
    #[error("mle not fully bound")]
    MleNotFullyBoundError,
    #[error("failed to bind variables during sumcheck")]
    SumcheckProverError,
    #[error("last round sumcheck fail")]
    SumcheckFinalFail,
    #[error("empty list for lhs or rhs")]
    EmptyMleList,
    #[error("bound indices fail to match challenge")]
    EvaluateBoundIndicesDontMatch,
    #[error("beta table associated is not indexed")]
    BetaTableNotIndexed,
}

/// evaluate_mle_ref_product without beta tables........
///
/// ---
///
/// Given (possibly half-fixed) bookkeeping tables of the MLEs which are multiplied,
/// e.g. V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// computes g_k(x) = \sum_{b_{k + 1}, ..., b_n} V_i(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n) * V_{i + 1}(u_1, ..., u_{k - 1}, x, b_{k + 1}, ..., b_n)
/// at `degree + 1` points.
///
/// ## Arguments
/// * `mle_refs` - MLEs pointing to the actual bookkeeping tables for the above
/// * `independent_variable` - whether the `x` from above resides within at least one of the `mle_refs`
/// * `degree` - degree of `g_k(x)`, i.e. number of evaluations to send (minus one!)
fn evaluate_mle_ref_product_gate<F: FieldExt>(
    mle_refs: &[impl MleRef<F = F>],
    independent_variable: bool,
    degree: usize,
) -> Result<Evals<F>, MleError> {
    for mle_ref in mle_refs {
        if !mle_ref.indexed() {
            return Err(MleError::NotIndexedError);
        }
    }
    // --- Gets the total number of iterated variables across all MLEs within this product ---
    let max_num_vars = mle_refs
        .iter()
        .map(|mle_ref| mle_ref.num_vars())
        .max()
        .ok_or(MleError::EmptyMleList)?;

    if independent_variable {
        // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
        let eval_count = degree + 1;

        // iterate across all pairs of evaluations
        let evals = cfg_into_iter!((0..1 << (max_num_vars - 1))).fold(
            #[cfg(feature = "parallel")]
            || vec![F::zero(); eval_count],
            #[cfg(not(feature = "parallel"))]
            vec![F::zero(); eval_count],
            |mut acc, index| {
                //get the product of all evaluations over 0/1/..degree
                let evals = mle_refs
                    .iter()
                    .map(|mle_ref| {
                        let zero = F::zero();
                        let index = if mle_ref.num_vars() < max_num_vars {
                            let max = 1 << mle_ref.num_vars();
                            (index * 2) % max
                        } else {
                            index * 2
                        };
                        let first = *mle_ref.bookkeeping_table().get(index).unwrap_or(&zero);
                        let second = if mle_ref.num_vars() != 0 {
                            *mle_ref.bookkeeping_table().get(index + 1).unwrap_or(&zero)
                        } else {
                            first
                        };

                        // let second = *mle_ref.mle().get(index + 1).unwrap_or(&zero);
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
            || vec![F::zero(); eval_count],
            |mut acc, partial| {
                acc.iter_mut()
                    .zip(partial)
                    .for_each(|(acc, partial)| *acc += partial);
                acc
            },
        );

        Ok(Evals(evals))
    } else {
        // There is no independent variable and we can sum over everything
        let sum = cfg_into_iter!((0..1 << (max_num_vars))).fold(
            #[cfg(feature = "parallel")]
            || F::zero(),
            #[cfg(not(feature = "parallel"))]
            F::zero(),
            |acc, index| {
                // Go through each MLE within the product
                let product = mle_refs
                    .iter()
                    // Result of this `map()`: A list of evaluations of the MLEs at `index`
                    .map(|mle_ref| {
                        let index = if mle_ref.num_vars() < max_num_vars {
                            // max = 2^{num_vars}; index := index % 2^{num_vars}
                            let max = 1 << mle_ref.num_vars();
                            index % max
                        } else {
                            index
                        };
                        // --- Access the MLE at that index. Pad with zeros ---
                        mle_ref
                            .bookkeeping_table()
                            .get(index)
                            .cloned()
                            .unwrap_or(F::zero())
                    })
                    .reduce(|acc, eval| acc * eval)
                    .unwrap();

                // --- Combine them into the accumulator ---
                // Note that the accumulator stores g(0), g(1), ..., g(d - 1)
                acc + product
            },
        );

        #[cfg(feature = "parallel")]
        let sum = sum.reduce(|| F::zero(), |acc, partial| acc + partial);

        Ok(Evals(vec![sum; degree]))
    }
}

/// checks whether mle was bound correctly to all the challenge points!!!!!!!!!!
pub fn check_fully_bound<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
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

            if indices != challenges {
                false
            } else {
                true
            }
        })
        .collect();

    if mles_bound.contains(&false) {
        return Err(GateError::EvaluateBoundIndicesDontMatch);
    }

    mle_refs.into_iter().fold(Ok(F::one()), |acc, mle_ref| {
        // --- Accumulate either errors or multiply ---
        let acc = acc?;
        if mle_ref.bookkeeping_table().len() != 1 {
            return Err(GateError::MleNotFullyBoundError);
        }
        Ok(acc * mle_ref.bookkeeping_table()[0])
    })
}

/// index mle indices for an array of mles
pub fn index_mle_indices_gate<F: FieldExt>(mle_refs: &mut [impl MleRef<F = F>], index: usize) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        mle_ref.index_mle_indices(index);
    })
}

/// fix variable for an array of mles
pub fn fix_var_gate<F: FieldExt>(
    mle_refs: &mut [impl MleRef<F = F>],
    round_index: usize,
    challenge: F,
) {
    mle_refs.iter_mut().for_each(|mle_ref| {
        if mle_ref
            .mle_indices()
            .contains(&MleIndex::IndexedBit(round_index))
        {
            mle_ref.fix_variable(round_index, challenge);
        }
    })
}

/// compute sumcheck message without a beta table!!!!!!!!!!!!!!
pub fn compute_sumcheck_message_add_gate<F: FieldExt>(
    lhs: &[impl MleRef<F = F>],
    rhs: &[impl MleRef<F = F>],
    round_index: usize,
) -> Result<Vec<F>, GateError> {
    // for gate mles, degree always 2 for left and right side because on each side we are taking the product of two bookkkeping tables
    let degree = 2;

    // --- Go through all of the MLEs being multiplied together on the LHS and see if any of them contain an IV ---
    // TODO!(ryancao): Should this not always be true...?
    let independent_variable_lhs = lhs
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let eval_lhs = evaluate_mle_ref_product_gate(lhs, independent_variable_lhs, degree).unwrap();

    // --- Similarly, but for the RHS ---
    let independent_variable_rhs = rhs
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let eval_rhs = evaluate_mle_ref_product_gate(rhs, independent_variable_rhs, degree).unwrap();

    // --- The evaluations of g_i(x) (i.e. the univariate sumcheck message) are simply the sum of those of the two sides ---
    let eval = eval_lhs + eval_rhs;

    let Evals(evaluations) = eval;

    Ok(evaluations)
}

/// Computes a round of the sumcheck protocol on this Layer
pub fn prove_round_add<F: FieldExt>(
    round_index: usize,
    challenge: F,
    lhs: &mut [impl MleRef<F = F>],
    rhs: &mut [impl MleRef<F = F>],
) -> Result<Vec<F>, GateError> {
    fix_var_gate(lhs, round_index - 1, challenge);
    fix_var_gate(rhs, round_index - 1, challenge);
    compute_sumcheck_message_add_gate(lhs, rhs, round_index)
}

/// computes the sumcheck message for batched gate mle
pub fn compute_sumcheck_message_copy_add<F: FieldExt>(
    beta: &mut BetaTable<F>,
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    round_index: usize,
) -> Result<Vec<F>, GateError> {
    // degree is 2 because we use a beta table
    let degree = 2;
    let independent_lhs = lhs
        .mle_indices()
        .contains(&MleIndex::IndexedBit(round_index));
    let independent_rhs = rhs
        .mle_indices()
        .contains(&MleIndex::IndexedBit(round_index));

    let evals_lhs =
        evaluate_mle_ref_product(&[lhs.clone()], independent_lhs, degree, beta.clone().table)
            .unwrap();
    let evals_rhs =
        evaluate_mle_ref_product(&[rhs.clone()], independent_rhs, degree, beta.clone().table)
            .unwrap();

    let eval = evals_lhs + evals_rhs;
    let Evals(evaluations) = eval;

    Ok(evaluations)
}

/// does all the necessary updates when proving a round for batched gate mles
pub fn prove_round_copy<F: FieldExt>(
    phase_lhs: &mut DenseMleRef<F>,
    phase_rhs: &mut DenseMleRef<F>,
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    beta: &mut BetaTable<F>,
    round_index: usize,
    challenge: F,
) -> Result<Vec<F>, GateError> {
    phase_lhs.fix_variable(round_index - 1, challenge);
    phase_rhs.fix_variable(round_index - 1, challenge);
    beta.beta_update(round_index - 1, challenge).unwrap();
    // need to separately update these because the phase_lhs and phase_rhs has no version of them
    lhs.fix_variable(round_index - 1, challenge);
    rhs.fix_variable(round_index - 1, challenge);
    compute_sumcheck_message_copy_add(beta, phase_lhs, phase_rhs, round_index)
}


/// fully evaluates a gate expression (for both the batched and non-batched case, add and mul gates)
pub fn compute_full_gate<F: FieldExt>(
    challenges: Vec<F>,
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    nonzero_gates: &Vec<(usize, usize, usize)>,
    copy_bits: usize,
) -> F {
    // split the challenges into which ones are for batched bits, which ones are for others
    let mut copy_chals: Vec<F> = vec![];
    let mut z_chals: Vec<F> = vec![];
    challenges.into_iter().enumerate().for_each(|(idx, chal)| {
        if (0..copy_bits).contains(&idx) {
            copy_chals.push(chal);
        } else {
            z_chals.push(chal);
        }
    });

    // if the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on z
    let beta_g = BetaTable::new((z_chals, F::zero())).unwrap();
    let zero = F::zero();

    // literally summing over everything else (x, y)
    if copy_bits == 0 {
        let sum =
            nonzero_gates
                .clone()
                .into_iter()
                .fold(F::zero(), |acc, (z_ind, x_ind, y_ind)| {
                    let gz = *beta_g
                        .table
                        .bookkeeping_table()
                        .get(z_ind)
                        .unwrap_or(&F::zero());
                    let ux = lhs.bookkeeping_table().get(x_ind).unwrap_or(&zero);
                    let vy = rhs.bookkeeping_table().get(y_ind).unwrap_or(&zero);
                    dbg!(&gz, ux, vy);
                    acc + gz * (*ux + *vy)
                });
        sum
    } else {
        let num_copy_idx = 1 << copy_bits;
        // if the gate looks like f1(z, x, y)(f2(p2, x) + f3(p2, y)) then this is the beta table for the challenges on p2
        let beta_g2 = BetaTable::new((copy_chals, F::zero())).unwrap();
        let sum = {
            // sum over everything else, outer sum being over p2, inner sum over (x, y)
            (0..(1 << num_copy_idx))
                .into_iter()
                .fold(F::zero(), |acc_outer, idx| {
                    let g2 = *beta_g2
                        .table
                        .bookkeeping_table()
                        .get(idx)
                        .unwrap_or(&F::zero());
                    let inner_sum = nonzero_gates.clone().into_iter().fold(
                        F::zero(),
                        |acc, (z_ind, x_ind, y_ind)| {
                            let gz = *beta_g
                                .table
                                .bookkeeping_table()
                                .get(z_ind)
                                .unwrap_or(&F::zero());
                            let ux = lhs
                                .bookkeeping_table()
                                .get(idx + (x_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            let vy = rhs
                                .bookkeeping_table()
                                .get(idx + (y_ind * num_copy_idx))
                                .unwrap_or(&zero);
                            dbg!(&gz, ux, vy);
                            acc + gz * (*ux + *vy)
                        },
                    );
                    acc_outer + (g2 * inner_sum)
                })
        };
        sum
    }
}

/// Computes a round of the sumcheck protocol on this Layer
pub fn prove_round_mul<F: FieldExt>(
    round_index: usize,
    challenge: F,
    mles: &mut [impl MleRef<F = F>],
) -> Result<Vec<F>, GateError> {
    fix_var_gate(mles, round_index - 1, challenge);
    compute_sumcheck_message_mul_gate(mles, round_index)
}

/// compute sumcheck message without a beta table!!!!!!!!!!!!!!
pub fn compute_sumcheck_message_mul_gate<F: FieldExt>(
    mles: &[impl MleRef<F = F>],
    round_index: usize,
) -> Result<Vec<F>, GateError> {
    // for gate mles, degree always 2 for left and right side because on each side we are taking the product of two bookkkeping tables
    let degree = 2;

    // --- Go through all of the MLEs being multiplied together on the LHS and see if any of them contain an IV ---
    // TODO!(ryancao): Should this not always be true...?
    let independent_variable = mles
        .iter()
        .map(|mle_ref| {
            mle_ref
                .mle_indices()
                .contains(&MleIndex::IndexedBit(round_index))
        })
        .reduce(|acc, item| acc | item)
        .ok_or(GateError::EmptyMleList)?;
    let eval = evaluate_mle_ref_product_gate(mles, independent_variable, degree).unwrap();

    let Evals(evaluations) = eval;

    Ok(evaluations)
}


/// does all the necessary updates when proving a round for batched gate mles
pub fn prove_round_copy_mul<F: FieldExt>(
    // phase_lhs: &mut DenseMleRef<F>,
    // phase_rhs: &mut DenseMleRef<F>,
    lhs: &mut DenseMleRef<F>,
    rhs: &mut DenseMleRef<F>,
    beta_g1: &BetaTable<F>,
    beta_g2: &mut BetaTable<F>,
    round_index: usize,
    challenge: F,
    nonzero_gates: &Vec<(usize, usize, usize)>,
    num_dataparallel_bits: usize,
) -> Result<Vec<F>, GateError> {
    // phase_lhs.fix_variable(round_index - 1, challenge);
    // phase_rhs.fix_variable(round_index - 1, challenge);
    beta_g2.beta_update(round_index - 1, challenge).unwrap();
    // need to separately update these because the phase_lhs and phase_rhs has no version of them
    lhs.fix_variable(round_index - 1, challenge);
    rhs.fix_variable(round_index - 1, challenge);
    // compute_sumcheck_message_copy_phase_mul(&[phase_lhs.clone(), phase_rhs.clone()], beta, round_index)
    libra_giraffe(&lhs, &rhs, &beta_g2.table, &beta_g1.table, nonzero_gates, num_dataparallel_bits)
}


/// get the evals for a batched mul gate 
pub fn libra_giraffe<F: FieldExt>(
    f2_p2_x: &DenseMleRef<F>,
    f3_p2_y: &DenseMleRef<F>,
    beta_g2: &DenseMleRef<F>,
    beta_g1: &DenseMleRef<F>,
    nonzero_gates: &Vec<(usize, usize, usize)>,
    num_dataparallel_bits: usize,
) -> Result<Vec<F>, GateError> {

    // always always ALWAYS 3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    // because we have a beta(g2, p2), f2(p2, x), and f3(p2, y)

    let degree = 3;

    if !beta_g2.indexed() {
        return Err(GateError::BetaTableNotIndexed);
    }

    // There is an independent variable, and we must extract `degree` evaluations of it, over `0..degree`
    let eval_count = degree + 1;

    // iterate across all pairs of evaluations
    let evals = cfg_into_iter!((0..1 << (num_dataparallel_bits - 1))).fold(
        #[cfg(feature = "parallel")]
        || vec![F::zero(); eval_count],
        #[cfg(not(feature = "parallel"))]
        vec![F::zero(); eval_count],
        |mut acc, p2_idx| {
            // compute the beta successors the same way it's done for each mle. do it outside the loop
            // because it only needs to be done once per product of mles
            let first = *beta_g2.bookkeeping_table().get(p2_idx * 2).unwrap();
            let second = if beta_g2.num_vars() != 0 {
                *beta_g2.bookkeeping_table().get(p2_idx * 2 + 1).unwrap()
            } else {
                first
            };
            let step = second - first;

            let beta_successors_snd =
                std::iter::successors(Some(second), move |item| Some(*item + step));
            //iterator that represents all evaluations of the MLE extended to arbitrarily many linear extrapolations on the line of 0/1
            let beta_successors = std::iter::once(first).chain(beta_successors_snd);
            let beta_iter: Box<dyn Iterator<Item = F>> = Box::new(beta_successors);


            let num_dataparallel_entries = 1 << num_dataparallel_bits;
            let inner_sum_successors = nonzero_gates.clone().into_iter().map(
                |(z, x, y)| {
                    let g1_z = *beta_g1.bookkeeping_table.get(z).unwrap();
                    let g1_z_successors = std::iter::successors(Some(g1_z), move |_| Some(g1_z));

                    // --- Compute f_2((A, p_2), x) ---
                    // --- Note that the bookkeeping table is little-endian, so we shift by `x * num_dataparallel_entries` ---
                    let f2_0_p2_x = *f2_p2_x.bookkeeping_table().get((p2_idx * 2) + x * num_dataparallel_entries).unwrap();
                    let f2_1_p2_x = if f2_p2_x.num_vars() != 0 {
                        *f2_p2_x.bookkeeping_table().get((p2_idx * 2 + 1) + x * num_dataparallel_entries).unwrap()
                    } else {
                        f2_0_p2_x
                    };
                    let linear_diff_f2 = f2_1_p2_x - f2_0_p2_x;
    
                    let f2_evals_p2_x =
                        std::iter::successors(Some(f2_1_p2_x), move |f2_prev_p2_x| Some(*f2_prev_p2_x + linear_diff_f2));
                    let all_f2_evals_p2_x = std::iter::once(f2_0_p2_x).chain(f2_evals_p2_x);

                    // --- Compute f_3((A, p_2), y) ---
                    // --- Note that the bookkeeping table is little-endian, so we shift by `y * num_dataparallel_entries` ---
                    let f3_0_p2_y = *f3_p2_y.bookkeeping_table().get((p2_idx * 2) + y * num_dataparallel_entries).unwrap();
                    let f3_1_p2_y = if f3_p2_y.num_vars() != 0 {
                        *f3_p2_y.bookkeeping_table().get((p2_idx * 2 + 1) + y * num_dataparallel_entries).unwrap()
                    } else {
                        f3_0_p2_y
                    };
                    let linear_diff_f3 = f3_1_p2_y - f3_0_p2_y;
    
                    let f3_evals_p2_y =
                        std::iter::successors(Some(f3_1_p2_y), move |f3_prev_p2_y| Some(*f3_prev_p2_y + linear_diff_f3));
                    let all_f3_evals_p2_y = std::iter::once(f3_0_p2_y).chain(f3_evals_p2_y);

                    // --- The evals we want are simply the element-wise product of the accessed evals ---
                    let g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y = g1_z_successors.zip(all_f2_evals_p2_x.zip(all_f3_evals_p2_y)).map(|(g1_z_eval, (f2_eval, f3_eval))| {
                        g1_z_eval * f2_eval * f3_eval
                    });

                    let evals_iter: Box<dyn Iterator<Item = F>> = Box::new(g1_z_times_f2_evals_p2_x_times_f3_evals_p2_y);

                    evals_iter
                }
            ).reduce(
                |acc, successor| {
                    let add_successors = acc.zip(successor).map(
                        |(acc_eval, successor_eval)| {
                            acc_eval + successor_eval
                        }
                    );

                    let add_iter: Box<dyn Iterator<Item = F>> = Box::new(add_successors);
                    add_iter
                }
            ).unwrap();


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
        || vec![F::zero(); eval_count],
        |mut acc, partial| {
            acc.iter_mut()
                .zip(partial)
                .for_each(|(acc, partial)| *acc += partial);
            acc
        },
    );
    Ok(evals)
}
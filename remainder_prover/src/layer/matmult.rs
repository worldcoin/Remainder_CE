//! This module contains the implementation of the matrix multiplication layer

use ::serde::{Deserialize, Serialize};
use ark_std::{cfg_into_iter, end_timer, log2, start_timer};
use itertools::Itertools;
use log::debug;
use ndarray::Array2;
use rand::Rng;
use remainder_shared_types::{
    claims::YieldClaim,
    layer::{sumcheck_layer::SumcheckLayer, Layer, LayerId},
    transcript::{ProverTranscript, VerifierTranscript},
    FieldExt,
};

use super::{
    combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs},
    gate::{check_fully_bound, compute_sumcheck_message_no_beta_table},
    product::{PostSumcheckLayer, Product},
    LayerError,
};
use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError,
    },
    layer::VerificationError,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_enum::MleEnum, Mle, MleIndex},
    prover::SumcheckProof,
    sumcheck::{evaluate_at_a_point, VerifyError},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Used to represent a matrix, along with its optional prefix bits (in circuit)
/// basically an equivalence of DenseMle<F>, but uninstantiated, until preprocessing
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct Matrix<F: FieldExt> {
    mle: MultilinearExtension<F>,
    num_rows_vars: usize,
    num_cols_vars: usize,
    prefix_bits: Option<Vec<bool>>,
    layer_id: Option<LayerId>,
}

impl<F: FieldExt> Matrix<F> {
    /// Create a new matrix, note that we require num_rows, and later converts this
    /// parameter to log(num_rows). This is necessary to check all dims are powers of 2
    pub fn new(
        mle: MultilinearExtension<F>,
        num_rows: usize,
        num_cols: usize,
        prefix_bits: Option<Vec<bool>>,
        layer_id: Option<LayerId>,
    ) -> Matrix<F> {
        assert_eq!(mle.get_evals_vector().len(), num_rows * num_cols);

        let mut new_bookkeeping_table = Vec::new();
        // pad the columns
        if 1 << log2(num_cols) != num_cols {
            let num_to_pad_each_row = (1 << log2(num_cols) as usize) - num_cols;
            for chunk in mle.get_evals_vector().chunks(num_cols) {
                new_bookkeeping_table.extend(
                    [chunk.to_vec(), vec![F::ZERO; num_to_pad_each_row]]
                        .into_iter()
                        .concat(),
                )
            }
        } else {
            new_bookkeeping_table = mle.get_evals_vector().to_vec();
        }

        // pad the rows
        let padded_matrix_len = (1 << log2(num_rows) as usize) * (1 << log2(num_cols) as usize);
        if new_bookkeeping_table.len() != padded_matrix_len {
            let num_need_to_pad = padded_matrix_len - new_bookkeeping_table.len();
            new_bookkeeping_table = [new_bookkeeping_table, vec![F::ZERO; num_need_to_pad]]
                .into_iter()
                .concat()
        }

        let mle = MultilinearExtension::new(new_bookkeeping_table);

        assert_eq!(padded_matrix_len, mle.get_evals_vector().len());

        Matrix {
            mle,
            num_rows_vars: log2(num_rows) as usize,
            num_cols_vars: log2(num_cols) as usize,
            prefix_bits,
            layer_id,
        }
    }

    /// get the dimension of this matrix
    pub fn num_vars_rows_cols(&self) -> (usize, usize) {
        (self.num_rows_vars, self.num_cols_vars)
    }
}

/// Used to represent a matrix multiplication layer
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct MatMult<F: FieldExt> {
    layer_id: LayerId,
    matrix_a: Matrix<F>,
    mle_a: Option<DenseMle<F>>,
    matrix_b: Matrix<F>,
    mle_b: Option<DenseMle<F>>,
    num_vars_middle_ab: Option<usize>,
}

impl<F: FieldExt> MatMult<F> {
    /// Create a new matrix multiplication layer
    pub fn new(layer_id: LayerId, matrix_a: Matrix<F>, matrix_b: Matrix<F>) -> MatMult<F> {
        MatMult {
            layer_id,
            matrix_a,
            mle_a: None,
            matrix_b,
            mle_b: None,
            num_vars_middle_ab: None,
        }
    }

    fn pre_processing_step(&mut self, claim_a: Vec<F>, claim_b: Vec<F>) {
        let matrix_a_mle = &mut self.matrix_a.mle;
        let matrix_b_mle = &mut self.matrix_b.mle;

        // check that both matrices are padded
        assert_eq!(
            (1 << self.matrix_a.num_cols_vars) * (1 << self.matrix_a.num_rows_vars),
            matrix_a_mle.get_evals_vector().len()
        );
        assert_eq!(
            (1 << self.matrix_b.num_cols_vars) * (1 << self.matrix_b.num_rows_vars),
            matrix_b_mle.get_evals_vector().len()
        );

        // check to make sure the dimensions match
        if self.matrix_a.num_cols_vars == self.matrix_b.num_rows_vars {
            self.num_vars_middle_ab = Some(self.matrix_a.num_cols_vars);
        } else {
            panic!("Matrix dimensions do not match")
        }

        let transpose_timer = start_timer!(|| "transpose matrix");
        let matrix_a_transp = gen_transpose_matrix(&self.matrix_a);
        end_timer!(transpose_timer);

        let mut matrix_a_transp = DenseMle::new_with_prefix_bits(
            matrix_a_transp.mle,
            matrix_a_transp.layer_id.unwrap(),
            matrix_a_transp.prefix_bits.unwrap_or(vec![]),
        );
        let mut matrix_b_mle = DenseMle::new_with_prefix_bits(
            self.matrix_b.mle.clone(),
            self.matrix_b.layer_id.unwrap(),
            self.matrix_b.prefix_bits.clone().unwrap_or(vec![]),
        );

        matrix_a_transp.index_mle_indices(0);
        matrix_b_mle.index_mle_indices(0);

        // bind the row indices of matrix a to relevant claim point
        claim_a.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_a_transp.fix_variable(idx, chal);
        });
        let mut bound_indices_a = vec![];

        let new_a_indices = matrix_a_transp
            .mle_indices
            .clone()
            .into_iter()
            .filter_map(|index: MleIndex<F>| {
                if let MleIndex::IndexedBit(_) = index {
                    Some(MleIndex::Iterated)
                } else if let MleIndex::Bound(..) = index {
                    bound_indices_a.push(index);
                    None
                } else {
                    Some(index)
                }
            })
            .collect_vec();

        let mut mle_a = DenseMle::new_from_raw(
            matrix_a_transp.bookkeeping_table().to_vec(),
            matrix_a_transp.layer_id,
        );

        mle_a.mle_indices = new_a_indices
            .into_iter()
            .chain(bound_indices_a.into_iter())
            .collect_vec();
        mle_a.index_mle_indices(0);

        self.mle_a = Some(mle_a);

        // bind the column indices of matrix b to relevant claim point
        claim_b.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_b_mle.fix_variable(idx, chal);
        });
        let new_b_indices = matrix_b_mle
            .clone()
            .mle_indices
            .into_iter()
            .map(|index| {
                if let MleIndex::IndexedBit(_) = index {
                    MleIndex::Iterated
                } else {
                    index
                }
            })
            .collect_vec();
        matrix_b_mle.mle_indices = new_b_indices;
        matrix_b_mle.index_mle_indices(0);

        self.mle_b = Some(matrix_b_mle)
    }

    /// Return evaluations of the univariate for one round of sumcheck with the matmult protocol, and then
    /// mutate the underlying bookkeeping tables with the current challenge.
    fn prove_round_matmul(&mut self, round: usize, challenge: F) -> Vec<F> {
        let mle_a = self.mle_a.as_mut().unwrap();
        let mle_b = self.mle_b.as_mut().unwrap();
        mle_a.fix_variable(round, challenge);
        mle_b.fix_variable(round, challenge);
        let next_message =
            compute_sumcheck_message_no_beta_table(&[mle_a.clone(), mle_b.clone()], 2, round)
                .unwrap();
        next_message
    }

    /// dummy sumcheck prover for this, testing purposes
    #[allow(dead_code)]
    fn dummy_prove_rounds(
        &mut self,
        claim: Claim<F>,
        rng: &mut impl Rng,
    ) -> Result<Vec<(Vec<F>, Option<F>)>, LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        self.pre_processing_step(claim_a, claim_b);

        let mut messages: Vec<(Vec<F>, Option<F>)> = vec![];
        let mut challenges: Vec<F> = vec![];
        let mut challenge: Option<F> = None;

        let first_message = compute_sumcheck_message_no_beta_table(
            &[self.mle_a.clone().unwrap(), self.mle_b.clone().unwrap()],
            0,
            2,
        )
        .unwrap();
        messages.push((first_message, challenge));

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        for round in 1..num_vars_middle {
            // TODO: raise error if None
            challenge = Some(F::from(rng.gen::<u64>()));
            let chal = challenge.unwrap();
            challenges.push(chal);
            self.mle_a
                .as_mut()
                .unwrap()
                .fix_variable(round - 1, challenge.clone().unwrap());
            self.mle_b
                .as_mut()
                .unwrap()
                .fix_variable(round - 1, challenge.clone().unwrap());
            let next_message = compute_sumcheck_message_no_beta_table(
                &[self.mle_a.clone().unwrap(), self.mle_b.clone().unwrap()],
                round,
                2,
            )
            .unwrap();
            messages.push((next_message, challenge));
        }

        Ok(messages)
    }

    /// dummy verifier for dummy sumcheck, testing purposes
    #[allow(dead_code)]
    fn dummy_verify_rounds(
        &mut self,
        messages: Vec<(Vec<F>, Option<F>)>,
        rng: &mut impl Rng,
        claim: Claim<F>,
    ) -> Result<(), VerifyError> {
        // first message check
        let mut prev_evals = &messages[0].0;
        let mut challenges = vec![];

        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let claimed_val = prev_evals[0] + prev_evals[1];
        if claimed_val != claim.get_result() {
            dbg!("hello");
            dbg!(messages[0].0[0] + messages[0].0[1]);
            return Err(VerifyError::SumcheckBad);
        }

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        // --- Go through sumcheck messages + (FS-generated) challenges ---
        for i in 1..num_vars_middle {
            // TODO: raise error if not
            let (evals, challenge) = &messages[i];
            let curr_evals = evals;
            let chal = (challenge).unwrap();
            // --- Evaluate the previous round's polynomial at the random challenge point, i.e. g_{i - 1}(r_i) ---
            let prev_at_r = evaluate_at_a_point(prev_evals, challenge.unwrap())
                .expect("could not evaluate at challenge point");

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                dbg!("whoops");
                dbg!(&prev_at_r);
                dbg!(curr_evals[0] + curr_evals[1]);
                return Err(VerifyError::SumcheckBad);
            };
            prev_evals = curr_evals;
            challenges.push(chal);
        }

        let final_chal = F::from(rng.gen::<u64>());
        challenges.push(final_chal);
        self.mle_a
            .as_mut()
            .unwrap()
            .fix_variable(num_vars_middle - 1, final_chal);
        self.mle_b
            .as_mut()
            .unwrap()
            .fix_variable(num_vars_middle - 1, final_chal);

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();

        let full_claim_chals_a = challenges
            .clone()
            .into_iter()
            .chain(claim_a.into_iter())
            .collect_vec();
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(challenges.into_iter())
            .collect_vec();

        let fully_bound_a =
            check_fully_bound(&mut [self.mle_a.clone().unwrap()], full_claim_chals_a).unwrap();
        let fully_bound_b =
            check_fully_bound(&mut [self.mle_b.clone().unwrap()], full_claim_chals_b).unwrap();
        let matrix_product = fully_bound_a * fully_bound_b;

        if prev_at_r != matrix_product {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }

    /// Return the PostSumcheckLayer, panicking if either of the MLE refs is not fully bound.
    pub fn get_post_sumcheck_layer(&self) -> PostSumcheckLayer<F, F> {
        let mle_refs = vec![self.mle_a.clone().unwrap(), self.mle_b.clone().unwrap()];
        PostSumcheckLayer(vec![Product::<F, F>::new(&mle_refs, F::ONE)])
    }
}

impl<F: FieldExt> Layer<F> for MatMult<F> {
    type Proof = Option<SumcheckProof<F>>;
    type Error = LayerError;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<Option<SumcheckProof<F>>, LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let pre_process_timer = start_timer!(|| "start preprocessing step");
        self.pre_processing_step(claim_a, claim_b);
        end_timer!(pre_process_timer);

        let mut challenges: Vec<F> = vec![];

        let first_message = compute_sumcheck_message_no_beta_table(
            &[self.mle_a.clone().unwrap(), self.mle_b.clone().unwrap()],
            0,
            2,
        )
        .unwrap();
        transcript_writer.append_elements("Initial Sumcheck evaluations", &first_message);

        let val = claim.get_result();
        if val != first_message[0] + first_message[1] {
            dbg!(&val);
            dbg!(first_message[0] + first_message[1]);
        }

        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        let sumcheck_rounds: Vec<Vec<F>> = std::iter::once(Ok(first_message))
            .chain((1..num_vars_middle).map(|round| {
                let challenge = transcript_writer.get_challenge("Sumcheck challenge");
                challenges.push(challenge);
                self.mle_a
                    .as_mut()
                    .unwrap()
                    .fix_variable(round - 1, challenge);
                self.mle_b
                    .as_mut()
                    .unwrap()
                    .fix_variable(round - 1, challenge);
                let next_message = compute_sumcheck_message_no_beta_table(
                    &[self.mle_a.clone().unwrap(), self.mle_b.clone().unwrap()],
                    round,
                    2,
                )
                .unwrap();

                transcript_writer.append_elements("Sumcheck evaluations", &next_message);
                Ok::<_, LayerError>(next_message)
            }))
            .try_collect()?;

        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
        challenges.push(final_chal);
        self.mle_a
            .as_mut()
            .unwrap()
            .fix_variable(num_vars_middle - 1, final_chal);
        self.mle_b
            .as_mut()
            .unwrap()
            .fix_variable(num_vars_middle - 1, final_chal);

        Ok(Some(sumcheck_rounds.into()))
    }

    /// Verifies the sumcheck protocol
    fn verify_rounds(
        &mut self,
        claim: Claim<F>,
        sumcheck_prover_messages: Self::Proof,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), LayerError> {
        let sumcheck_prover_messages = sumcheck_prover_messages.unwrap().0;
        let mut challenges = vec![];

        let mut prev_evals = &sumcheck_prover_messages[0];
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        let claimed_val = prev_evals[0] + prev_evals[1];

        if claimed_val != claim.get_result() {
            debug!("I'm the PROBLEM");
            debug!("msg0 + msg1 =\n{:?}", prev_evals[0] + prev_evals[1]);
            debug!("rest =\n{:?}", claim.get_result());
            return Err(LayerError::VerificationError(
                VerificationError::SumcheckStartFailed,
            ));
        }

        let num_prev_evals = sumcheck_prover_messages[0].len();
        transcript_reader
            .consume_elements("Initial Sumcheck evaluations", num_prev_evals)
            .unwrap();

        // --- For round 1 < i < n, perform the check ---
        // g_{i - 1}(r_i) = g_i(0) + g_i(1)
        for curr_evals in sumcheck_prover_messages.iter().skip(1) {
            let challenge = transcript_reader
                .get_challenge("Sumcheck challenge")
                .unwrap();

            let prev_at_r =
                evaluate_at_a_point(prev_evals, challenge).map_err(LayerError::InterpError)?;

            if prev_at_r != curr_evals[0] + curr_evals[1] {
                return Err(LayerError::VerificationError(
                    VerificationError::SumcheckFailed,
                ));
            };

            transcript_reader
                .consume_elements("Sumcheck evaluations", curr_evals.len())
                .unwrap();

            prev_evals = curr_evals;
            challenges.push(challenge);
        }

        // --- In the final round, we check that g(r_1, ..., r_n) = g_n(r_n) ---
        // Here, we first sample r_n.
        let final_chal = transcript_reader
            .get_challenge("Final Sumcheck challenge")
            .unwrap();
        challenges.push(final_chal);

        let prev_at_r = evaluate_at_a_point(prev_evals, final_chal).unwrap();
        let full_claim_chals_a = challenges
            .clone()
            .into_iter()
            .chain(claim_a.into_iter())
            .collect_vec();
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(challenges.into_iter())
            .collect_vec();
        let fully_bound_a =
            check_fully_bound(&mut [self.mle_a.clone().unwrap()], full_claim_chals_a).unwrap();
        let fully_bound_b =
            check_fully_bound(&mut [self.mle_b.clone().unwrap()], full_claim_chals_b).unwrap();
        let matrix_product = fully_bound_a * fully_bound_b;

        if prev_at_r != matrix_product {
            return Err(LayerError::VerificationError(
                VerificationError::FinalSumcheckFailed,
            ));
        }

        Ok(())
    }

    ///Gets this layers id
    fn id(&self) -> &LayerId {
        &self.layer_id
    }
}

impl<F: FieldExt> SumcheckLayer<F> for MatMult<F> {
    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), Self::Error> {
        let mut claim_b = claim_point.to_vec();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);
        self.pre_processing_step(claim_a, claim_b);
        Ok(())
    }

    fn prove_sumcheck_round(
        &mut self,
        round_index: usize,
        challenge: F,
    ) -> Result<Vec<F>, Self::Error> {
        Ok(self.prove_round_matmul(round_index, challenge))
    }

    fn num_sumcheck_rounds(&self) -> usize {
        self.num_vars_middle_ab.unwrap()
    }
}

impl<F: FieldExt> YieldClaim<ClaimMle<F>> for MatMult<F> {
    type Error = LayerError;
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let claims = vec![&self.mle_a, &self.mle_b]
            .into_iter()
            .map(|matrix| {
                let matrix_fixed_indices = matrix
                    .as_ref()
                    .unwrap()
                    .mle_indices()
                    .into_iter()
                    .map(|index| {
                        index
                            .val()
                            .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))
                            .unwrap()
                    })
                    .collect_vec();

                let matrix_val = matrix.as_ref().unwrap().bookkeeping_table()[0];
                let claim: ClaimMle<F> = ClaimMle::new(
                    matrix_fixed_indices,
                    matrix_val,
                    Some(self.id().clone()),
                    Some(matrix.as_ref().unwrap().layer_id),
                    Some(MleEnum::Dense(matrix.clone().unwrap())),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for MatMult<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claim_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);

        let mut claim_mles = claim_mles.clone();

        if let Some(common_idx) = common_idx {
            pre_fix_mle_refs(&mut claim_mles, &claim_vecs[0], common_idx);
        }

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                let wlx_eval_on_mle_ref = combine_mle_refs_with_aggregate(&claim_mles, &new_chal);
                wlx_eval_on_mle_ref.unwrap()
            })
            .collect();

        // concat this with the first k evaluations from the claims to
        // get num_evals evaluations
        let mut wlx_evals = claimed_vals.to_vec();
        wlx_evals.extend(&next_evals);
        Ok(wlx_evals)
    }
}

impl<F: std::fmt::Debug + FieldExt> MatMult<F> {
    pub(crate) fn circuit_description_fmt<'a>(&'a self) -> impl std::fmt::Display + 'a {
        // --- Dummy struct which simply exists to implement `std::fmt::Display` ---
        // --- so that it can be returned as an `impl std::fmt::Display` ---
        struct MatMultCircuitDesc<'a, F: std::fmt::Debug + FieldExt>(&'a MatMult<F>);

        impl<'a, F: std::fmt::Debug + FieldExt> std::fmt::Display for MatMultCircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("MatMult")
                    .field(
                        "matrix_a_layer_id",
                        &self.0.mle_a.as_ref().unwrap().layer_id,
                    )
                    .field(
                        "matrix_a_mle_indices",
                        &self.0.mle_a.as_ref().unwrap().mle_indices,
                    )
                    .field(
                        "matrix_b_layer_id",
                        &self.0.mle_b.as_ref().unwrap().layer_id,
                    )
                    .field(
                        "matrix_b_mle_indices",
                        &self.0.mle_b.as_ref().unwrap().mle_indices,
                    )
                    .field("num_vars_middle_ab", &self.0.matrix_a.num_cols_vars)
                    .finish()
            }
        }
        MatMultCircuitDesc(self)
    }
}

/// Generate the transpose of a matrix, uses Array2 from ndarray
pub fn gen_transpose_matrix<F: FieldExt>(matrix: &Matrix<F>) -> Matrix<F> {
    let num_rows = 1 << matrix.num_rows_vars;
    let num_cols = 1 << matrix.num_cols_vars;

    let matrix_array_2 =
        Array2::from_shape_vec((num_rows, num_cols), matrix.mle.get_evals_vector().to_vec())
            .unwrap();
    let matrix_transpose = matrix_array_2.reversed_axes();
    let matrix_transp_vec = matrix_transpose
        .outer_iter()
        .map(|x| x.to_vec())
        .flat_map(|row| row)
        .collect_vec();

    let mle = MultilinearExtension::new(matrix_transp_vec);

    Matrix::new(
        mle,
        num_rows,
        num_cols,
        matrix.prefix_bits.clone(),
        matrix.layer_id.clone(),
    )
}

/// Multiply two matrices together, with a transposed matrix_b
pub fn product_two_matrices<F: FieldExt>(matrix_a: &Matrix<F>, matrix_b: &Matrix<F>) -> Vec<F> {
    let num_middle_ab = 1 << matrix_a.num_cols_vars;

    let matrix_b_transpose = gen_transpose_matrix(matrix_b);

    let product_matrix = matrix_a
        .mle
        .get_evals_vector()
        .chunks(num_middle_ab as usize)
        .flat_map(|chunk_a| {
            matrix_b_transpose
                .mle
                .get_evals_vector()
                .chunks(num_middle_ab)
                .map(|chunk_b| {
                    chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .fold(F::ZERO, |acc, (&a, &b)| acc + (a * b))
                })
                .collect_vec()
        })
        .collect_vec();

    product_matrix
}

#[cfg(test)]
mod test {
    use ark_std::test_rng;
    use remainder_shared_types::{layer::LayerId, Fr};

    use crate::{
        claims::Claim,
        layer::matmult::{product_two_matrices, MatMult, Matrix},
        mle::{dense::DenseMle, evals::MultilinearExtension, Mle},
    };

    #[test]
    fn test_product_two_matrices() {
        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
        ];
        let mle_vec_b = vec![Fr::from(3), Fr::from(5), Fr::from(9), Fr::from(6)];

        let matrix_a = Matrix::new(MultilinearExtension::new(mle_vec_a), 4, 2, None, None);
        let matrix_b = Matrix::new(MultilinearExtension::new(mle_vec_b), 2, 2, None, None);

        let res_product = product_two_matrices(&matrix_a, &matrix_b);

        let exp_product = vec![
            Fr::from(1 * 3 + 2 * 9),
            Fr::from(1 * 5 + 2 * 6),
            Fr::from(9 * 3 + 10 * 9),
            Fr::from(9 * 5 + 10 * 6),
            Fr::from(13 * 3 + 1 * 9),
            Fr::from(13 * 5 + 1 * 6),
            Fr::from(3 * 3 + 10 * 9),
            Fr::from(3 * 5 + 10 * 6),
        ];

        assert_eq!(res_product, exp_product);
    }

    #[test]
    fn test_product_two_matrices_2() {
        let mle_vec_a = vec![
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(2),
            Fr::from(9),
            Fr::from(0),
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(4),
            Fr::from(2),
            Fr::from(4),
            Fr::from(2),
            Fr::from(6),
            Fr::from(7),
            Fr::from(3),
            Fr::from(4),
            Fr::from(1),
            Fr::from(6),
            Fr::from(2),
            Fr::from(9),
            Fr::from(0),
            Fr::from(1),
            Fr::from(4),
            Fr::from(5),
            Fr::from(4),
            Fr::from(2),
            Fr::from(4),
            Fr::from(2),
            Fr::from(6),
            Fr::from(7),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(2),
            Fr::from(1),
            Fr::from(5),
            Fr::from(3),
            Fr::from(6),
            Fr::from(7),
            Fr::from(4),
        ];

        let matrix_a = Matrix::new(MultilinearExtension::new(mle_vec_a), 8, 4, None, None);
        let matrix_b = Matrix::new(MultilinearExtension::new(mle_vec_b), 4, 2, None, None);

        let res_product = product_two_matrices(&matrix_a, &matrix_b);

        let exp_product = vec![
            Fr::from(58),
            Fr::from(56),
            Fr::from(22),
            Fr::from(53),
            Fr::from(43),
            Fr::from(65),
            Fr::from(81),
            Fr::from(82),
            Fr::from(58),
            Fr::from(56),
            Fr::from(22),
            Fr::from(53),
            Fr::from(43),
            Fr::from(65),
            Fr::from(81),
            Fr::from(82),
        ];

        assert_eq!(res_product, exp_product);
    }

    #[test]
    fn test_product_irregular_matrices() {
        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(1),
            Fr::from(3),
        ];

        let matrix_a = Matrix::new(MultilinearExtension::new(mle_vec_a), 5, 3, None, None);
        let matrix_b = Matrix::new(MultilinearExtension::new(mle_vec_b), 3, 3, None, None);

        let res_product = product_two_matrices(&matrix_a, &matrix_b);

        // 1  2  9
        // 10 13 1       3  5  9
        // 3  10 2   `   6  5  9
        // 9  10 1       6  1  3
        // 3  10 2

        let exp_product = vec![
            Fr::from(1 * 3 + 2 * 6 + 9 * 6),
            Fr::from(1 * 5 + 2 * 5 + 9 * 1),
            Fr::from(1 * 9 + 2 * 9 + 9 * 3),
            Fr::from(10 * 3 + 13 * 6 + 1 * 6),
            Fr::from(10 * 5 + 13 * 5 + 1 * 1),
            Fr::from(10 * 9 + 13 * 9 + 1 * 3),
            Fr::from(3 * 3 + 10 * 6 + 2 * 6),
            Fr::from(3 * 5 + 10 * 5 + 2 * 1),
            Fr::from(3 * 9 + 10 * 9 + 2 * 3),
            Fr::from(9 * 3 + 10 * 6 + 1 * 6),
            Fr::from(9 * 5 + 10 * 5 + 1 * 1),
            Fr::from(9 * 9 + 10 * 9 + 1 * 3),
            Fr::from(3 * 3 + 10 * 6 + 2 * 6),
            Fr::from(3 * 5 + 10 * 5 + 2 * 1),
            Fr::from(3 * 9 + 10 * 9 + 2 * 3),
        ];

        let matrix_out = Matrix::new(MultilinearExtension::new(exp_product), 5, 3, None, None);

        assert_eq!(res_product, matrix_out.mle.get_evals_vector().clone());
    }

    #[test]
    /// super basic symmetric test
    fn test_sumcheck_1() {
        let mut rng = test_rng();
        let claim = Claim::new(vec![Fr::from(1), Fr::from(0)], Fr::from(3));

        let matrix_a_vec = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let matrix_b_vec = vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(1)];

        let matrix_a: Matrix<Fr> = Matrix::new(
            MultilinearExtension::new(matrix_a_vec),
            2,
            2,
            None,
            Some(LayerId::Input(0)),
        );
        let matrix_b: Matrix<Fr> = Matrix::new(
            MultilinearExtension::new(matrix_b_vec),
            2,
            2,
            None,
            Some(LayerId::Input(0)),
        );

        let mut matrix_init: MatMult<Fr> = MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }

    #[test]
    /// super basic asymmetric test
    fn test_sumcheck_asymmetric() {
        let mut rng = test_rng();
        let claim = Claim::new(vec![Fr::from(1)], Fr::from(8));

        let matrix_a_vec = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(3),
            Fr::from(1),
            Fr::from(3),
            Fr::from(1),
            Fr::from(1),
            Fr::from(2),
        ];
        let matrix_b_vec = vec![Fr::from(1), Fr::from(2), Fr::from(1), Fr::from(1)];

        let matrix_a: Matrix<Fr> = Matrix::new(
            MultilinearExtension::new(matrix_a_vec),
            2,
            4,
            None,
            Some(LayerId::Input(0)),
        );
        let matrix_b: Matrix<Fr> = Matrix::new(
            MultilinearExtension::new(matrix_b_vec),
            4,
            1,
            None,
            Some(LayerId::Input(0)),
        );

        let mut matrix_init: MatMult<Fr> = MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }

    #[test]
    fn test_sumcheck_irregular_matrices() {
        let mut rng = test_rng();

        let mle_vec_a = vec![
            Fr::from(1),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(13),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
            Fr::from(9),
            Fr::from(10),
            Fr::from(1),
            Fr::from(3),
            Fr::from(10),
            Fr::from(2),
        ];
        let mle_vec_b = vec![
            Fr::from(3),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(5),
            Fr::from(9),
            Fr::from(6),
            Fr::from(1),
            Fr::from(3),
        ];

        let matrix_a = Matrix::new(
            MultilinearExtension::new(mle_vec_a),
            5,
            3,
            None,
            Some(LayerId::Input(0)),
        );
        let matrix_b = Matrix::new(
            MultilinearExtension::new(mle_vec_b),
            3,
            3,
            None,
            Some(LayerId::Input(0)),
        );

        let res_product = product_two_matrices(&matrix_a, &matrix_b);
        let mut mle_product_ref = DenseMle::new_from_raw(res_product, LayerId::Input(0));

        let _ = mle_product_ref.index_mle_indices(0);

        mle_product_ref.fix_variable(0, Fr::from(1));
        mle_product_ref.fix_variable(1, Fr::from(2));
        mle_product_ref.fix_variable(2, Fr::from(3));
        mle_product_ref.fix_variable(3, Fr::from(4));
        mle_product_ref.fix_variable(4, Fr::from(5));

        assert_eq!(mle_product_ref.bookkeeping_table().len(), 1);

        let claim = Claim::new(
            vec![
                Fr::from(1),
                Fr::from(2),
                Fr::from(3),
                Fr::from(4),
                Fr::from(5),
            ],
            mle_product_ref.bookkeeping_table()[0],
        );

        let mut matrix_init: MatMult<Fr> = MatMult::new(LayerId::Input(0), matrix_a, matrix_b);

        let messages = matrix_init.dummy_prove_rounds(claim.clone(), &mut rng);
        let verify_res = matrix_init.dummy_verify_rounds(messages.unwrap(), &mut rng, claim);

        assert!(verify_res.is_ok());
    }
}

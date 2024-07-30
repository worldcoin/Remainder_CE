//! This module contains the implementation of the matrix multiplication layer

use ::serde::{Deserialize, Serialize};
use ark_std::{cfg_into_iter, end_timer, log2, start_timer};
use itertools::Itertools;
use ndarray::Array2;
use rand::Rng;
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};

use super::{
    combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs},
    gate::{check_fully_bound, compute_sumcheck_message_no_beta_table},
    CircuitLayer, Layer, LayerError, LayerId, VerifierLayer,
};
use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::{circuit_expr::CircuitMle, verifier_expr::VerifierMle},
    layer::VerificationError,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_enum::MleEnum, Mle, MleIndex},
    sumcheck::{evaluate_at_a_point, VerifyError},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Used to represent a matrix, along with its optional prefix bits (in circuit)
/// basically an equivalence of DenseMle<F>, but uninstantiated, until preprocessing
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: FieldExt")]
pub struct Matrix<F: FieldExt> {
    mle: DenseMle<F>,
    num_rows_vars: usize,
    num_cols_vars: usize,
}

impl<F: FieldExt> Matrix<F> {
    /// Create a new matrix, note that we require num_rows, and later converts this
    /// parameter to log(num_rows). This is necessary to check all dims are powers of 2
    pub fn new(mle: DenseMle<F>, num_rows: usize, num_cols: usize) -> Matrix<F> {
        assert_eq!(mle.bookkeeping_table().len(), num_rows * num_cols);

        let mut new_bookkeeping_table = Vec::new();
        // pad the columns
        if 1 << log2(num_cols) != num_cols {
            assert!((1 << log2(num_cols) as usize) > num_cols);
            let num_to_pad_each_row = (1 << log2(num_cols) as usize) - num_cols;
            for chunk in mle.bookkeeping_table().chunks(num_cols) {
                new_bookkeeping_table.extend(
                    [chunk.to_vec(), vec![F::ZERO; num_to_pad_each_row]]
                        .into_iter()
                        .concat(),
                )
            }
        } else {
            new_bookkeeping_table = mle.bookkeeping_table().to_vec();
        }

        // pad the rows
        let padded_matrix_len = (1 << log2(num_rows) as usize) * (1 << log2(num_cols) as usize);
        if new_bookkeeping_table.len() != padded_matrix_len {
            assert!((1 << log2(num_rows) as usize) > num_rows);
            let num_need_to_pad = padded_matrix_len - new_bookkeeping_table.len();
            new_bookkeeping_table = [new_bookkeeping_table, vec![F::ZERO; num_need_to_pad]]
                .into_iter()
                .concat()
        }

        // pad the MLE indices as well!
        let expected_num_iterated_vars = (log2(num_rows) + log2(num_cols)) as usize;
        let new_indices = if mle.num_iterated_vars() != expected_num_iterated_vars {
            assert!(expected_num_iterated_vars > mle.num_iterated_vars());
            let num_iterated_vars_to_add = expected_num_iterated_vars - mle.num_iterated_vars();
            let padding_indices = vec![MleIndex::Iterated; num_iterated_vars_to_add];
            &[mle.mle_indices.to_vec(), padding_indices].concat()
        } else {
            mle.mle_indices()
        };

        let mle = DenseMle::new_with_indices(&new_bookkeeping_table, mle.layer_id(), new_indices);

        assert_eq!(padded_matrix_len, mle.bookkeeping_table().len());

        Matrix {
            mle,
            num_rows_vars: log2(num_rows) as usize,
            num_cols_vars: log2(num_cols) as usize,
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
    matrix_b: Matrix<F>,
    num_vars_middle_ab: Option<usize>,
}

impl<F: FieldExt> MatMult<F> {
    /// Create a new matrix multiplication layer
    pub fn new(layer_id: LayerId, matrix_a: Matrix<F>, matrix_b: Matrix<F>) -> MatMult<F> {
        MatMult {
            layer_id,
            matrix_a,
            matrix_b,
            num_vars_middle_ab: None,
        }
    }

    fn pre_processing_step(&mut self, claim_a: Vec<F>, claim_b: Vec<F>) {
        let matrix_a_mle = &mut self.matrix_a.mle;
        let matrix_b_mle = &mut self.matrix_b.mle;

        // check that both matrices are padded
        assert_eq!(
            (1 << self.matrix_a.num_cols_vars) * (1 << self.matrix_a.num_rows_vars),
            matrix_a_mle.bookkeeping_table().len()
        );
        assert_eq!(
            (1 << self.matrix_b.num_cols_vars) * (1 << self.matrix_b.num_rows_vars),
            matrix_b_mle.bookkeeping_table().len()
        );

        // check to make sure the dimensions match
        if self.matrix_a.num_cols_vars == self.matrix_b.num_rows_vars {
            self.num_vars_middle_ab = Some(self.matrix_a.num_cols_vars);
        } else {
            panic!("Matrix dimensions do not match")
        }

        let transpose_timer = start_timer!(|| "transpose matrix");
        let mut matrix_a_transp = gen_transpose_matrix(&self.matrix_a);
        end_timer!(transpose_timer);

        let mut matrix_a_transp = matrix_a_transp.mle;

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

        self.matrix_a.mle = mle_a;

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
            &[self.matrix_a.mle.clone(), self.matrix_b.mle.clone()],
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
            self.matrix_a
                .mle
                .fix_variable(round - 1, challenge.clone().unwrap());
            self.matrix_b
                .mle
                .fix_variable(round - 1, challenge.clone().unwrap());
            let next_message = compute_sumcheck_message_no_beta_table(
                &[self.matrix_a.mle.clone(), self.matrix_b.mle.clone()],
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
        self.matrix_a
            .mle
            .fix_variable(num_vars_middle - 1, final_chal);
        self.matrix_b
            .mle
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
            check_fully_bound(&mut [self.matrix_a.mle.clone()], full_claim_chals_a).unwrap();
        let fully_bound_b =
            check_fully_bound(&mut [self.matrix_b.mle.clone()], full_claim_chals_b).unwrap();
        let matrix_product = fully_bound_a * fully_bound_b;

        if prev_at_r != matrix_product {
            return Err(VerifyError::SumcheckBad);
        }

        Ok(())
    }

    fn append_leaf_mles_to_transcript(
        &self,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        assert_eq!(self.matrix_a.mle.bookkeeping_table().len(), 1);
        assert_eq!(self.matrix_b.mle.bookkeeping_table().len(), 1);

        transcript_writer.append_elements(
            "Fully bound matrix evaluations",
            &[
                self.matrix_a.mle.bookkeeping_table()[0],
                self.matrix_b.mle.bookkeeping_table()[0],
            ],
        );
    }
}

impl<F: FieldExt> Layer<F> for MatMult<F> {
    // type Proof = Option<SumcheckProof<F>>;
    type CircuitLayer = CircuitMatMultLayer<F>;

    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) -> Result<(), LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);
        self.pre_processing_step(claim_a, claim_b);

        let mut challenges: Vec<F> = vec![];
        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        for round in 0..num_vars_middle {
            let challenge = transcript_writer.get_challenge("Sumcheck challenge");
            challenges.push(challenge);
            self.matrix_a.mle.fix_variable(round - 1, challenge);
            self.matrix_b.mle.fix_variable(round - 1, challenge);
            let next_message = compute_sumcheck_message_no_beta_table(
                &[self.matrix_a.mle.clone(), self.matrix_b.mle.clone()],
                round,
                2,
            )
            .unwrap();
            transcript_writer.append_elements("Sumcheck evaluations", &next_message);
        }

        let final_chal = transcript_writer.get_challenge("Final Sumcheck challenge for binding x");
        challenges.push(final_chal);
        self.matrix_a
            .mle
            .fix_variable(num_vars_middle - 1, final_chal);
        self.matrix_b
            .mle
            .fix_variable(num_vars_middle - 1, final_chal);

        self.append_leaf_mles_to_transcript(transcript_writer);
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn into_circuit_layer(&self) -> Result<Self::CircuitLayer, LayerError> {
        Ok(self.clone().into())
    }
}

/// The circuit description counterpart of a [Matrix].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct CircuitMatrix<F: FieldExt> {
    mle: CircuitMle<F>,
    num_rows_vars: usize,
    num_cols_vars: usize,
}

impl<F: FieldExt> From<Matrix<F>> for CircuitMatrix<F> {
    fn from(matrix: Matrix<F>) -> Self {
        let mut indexed_mle = matrix.mle.clone();
        indexed_mle.index_mle_indices(0);
        CircuitMatrix {
            mle: CircuitMle::from_dense_mle(&indexed_mle).unwrap(),
            num_rows_vars: matrix.num_rows_vars,
            num_cols_vars: matrix.num_cols_vars,
        }
    }
}
/// The circuit description counterpart of a [MatMult] layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct CircuitMatMultLayer<F: FieldExt> {
    /// The layer id associated with this matmult layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: CircuitMatrix<F>,

    /// The RHS Matrix to be multiplied.
    matrix_b: CircuitMatrix<F>,
}

impl<F: FieldExt> From<MatMult<F>> for CircuitMatMultLayer<F> {
    /// Convert a [MatMult] to a [CircuitMatmultLayer].
    fn from(matmult_layer: MatMult<F>) -> Self {
        CircuitMatMultLayer {
            layer_id: matmult_layer.layer_id,
            matrix_a: matmult_layer.matrix_a.into(),
            matrix_b: matmult_layer.matrix_b.into(),
        }
    }
}

impl<F: FieldExt> CircuitMatMultLayer<F> {
    /// Convert a [CircuitMatMultLayer] to a [VerifierMatMultLayer], which represents a fully bound MatMult layer.
    pub fn into_verifier_matmult_layer(
        &self,
        point_a: &[F],
        point_b: &[F],
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> VerifierMatMultLayer<F> {
        let matrix_a = VerifierMatrix {
            mle: self
                .matrix_a
                .mle
                .into_verifier_mle(point_a, transcript_reader)
                .unwrap(),
            num_rows_vars: self.matrix_a.num_rows_vars,
            num_cols_vars: self.matrix_a.num_cols_vars,
        };

        let matrix_b = VerifierMatrix {
            mle: self
                .matrix_b
                .mle
                .into_verifier_mle(point_b, transcript_reader)
                .unwrap(),
            num_rows_vars: self.matrix_b.num_rows_vars,
            num_cols_vars: self.matrix_b.num_cols_vars,
        };

        VerifierMatMultLayer {
            layer_id: self.layer_id,
            matrix_a,
            matrix_b,
        }
    }
}

impl<F: FieldExt> CircuitLayer<F> for CircuitMatMultLayer<F> {
    type VerifierLayer = VerifierMatMultLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        // First split the claim made on this layer to the according points that are bound to
        // matrix A and matrix B.
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.num_cols_vars);

        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
        let mut g_prev_round = vec![claim.get_result()];

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = F::ZERO;

        // Get the number of rounds, which is exactly the inner dimension of the matrix product.
        assert_eq!(self.matrix_a.num_cols_vars, self.matrix_b.num_rows_vars);
        let num_rounds = self.matrix_a.num_cols_vars;

        // For round 1 <= i <= n, perform the check:
        for _round in 0..num_rounds {
            let degree = 2;

            let g_cur_round = transcript_reader
                .consume_elements("Sumcheck message", degree + 1)
                .map_err(|err| VerificationError::TranscriptError(err))?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_challenge("Sumcheck challenge")?;

            // Verify that:
            //       `g_i(0) + g_i(1) == g_{i - 1}(r_{i-1})`
            let g_i_zero = evaluate_at_a_point(&g_cur_round, F::ZERO).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, F::ONE).unwrap();
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();

            if g_i_zero + g_i_one != g_prev_r_prev {
                return Err(VerificationError::SumcheckFailed);
            }

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // Evalute `g_n(r_n)`.
        // Note: If there were no nonlinear rounds, this value reduces to
        // `claim.get_result()` due to how we initialized `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        let full_claim_chals_a = challenges
            .clone()
            .into_iter()
            .chain(claim_a.into_iter())
            .collect_vec();
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(challenges.into_iter())
            .collect_vec();

        let verifier_layer: VerifierMatMultLayer<F> = self.into_verifier_matmult_layer(
            &full_claim_chals_a,
            &full_claim_chals_b,
            transcript_reader,
        );

        let matrix_product = verifier_layer.evaluate();

        if g_final_r_final != matrix_product {
            return Err(VerificationError::FinalSumcheckFailed);
        }

        Ok(verifier_layer)
    }
}

/// The verifier's counterpart of a [Matrix].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMatrix<F: FieldExt> {
    mle: VerifierMle<F>,
    num_rows_vars: usize,
    num_cols_vars: usize,
}

/// The verifier's counterpart of a [MatMult] layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: FieldExt")]
pub struct VerifierMatMultLayer<F: FieldExt> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: VerifierMatrix<F>,

    /// The RHS Matrix to be multiplied.
    matrix_b: VerifierMatrix<F>,
}

impl<F: FieldExt> VerifierLayer<F> for VerifierMatMultLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: FieldExt> VerifierMatMultLayer<F> {
    fn evaluate(&self) -> F {
        self.matrix_a.mle.value() * self.matrix_b.mle.value()
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for VerifierMatMultLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let claims = vec![&self.matrix_a, &self.matrix_b]
            .into_iter()
            .map(|matrix| {
                let matrix_fixed_indices = matrix
                    .mle
                    .mle_indices()
                    .into_iter()
                    .map(|index| {
                        index
                            .val()
                            .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))
                            .unwrap()
                    })
                    .collect_vec();

                let mle_layer_id = matrix.mle.layer_id();
                let matrix_claimed_val = matrix.mle.value();

                // Dummy MLE ref.
                // TODO(ryancao): Fix things so that we don't need to pass this around... This is not right
                let mle_ref = MleEnum::Dense(DenseMle::new_from_raw(
                    vec![matrix_claimed_val],
                    mle_layer_id,
                ));

                let claim: ClaimMle<F> = ClaimMle::new(
                    matrix_fixed_indices,
                    matrix_claimed_val,
                    Some(self.layer_id.clone()),
                    Some(matrix.mle.layer_id()),
                    Some(mle_ref),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

impl<F: FieldExt> YieldClaim<F, ClaimMle<F>> for MatMult<F> {
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let claims = vec![&self.matrix_a.mle, &self.matrix_b.mle]
            .into_iter()
            .map(|matrix_mle| {
                let matrix_fixed_indices = matrix_mle
                    .mle_indices()
                    .into_iter()
                    .map(|index| {
                        index
                            .val()
                            .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))
                            .unwrap()
                    })
                    .collect_vec();

                let matrix_val = matrix_mle.bookkeeping_table()[0];
                let claim: ClaimMle<F> = ClaimMle::new(
                    matrix_fixed_indices,
                    matrix_val,
                    Some(self.layer_id.clone()),
                    Some(matrix_mle.layer_id),
                    Some(MleEnum::Dense(matrix_mle.clone())),
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
                    .field("matrix_a_layer_id", &self.0.matrix_a.mle.layer_id)
                    .field("matrix_a_mle_indices", &self.0.matrix_a.mle.mle_indices)
                    .field("matrix_b_layer_id", &self.0.matrix_b.mle.layer_id)
                    .field("matrix_b_mle_indices", &self.0.matrix_b.mle.mle_indices)
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

    let matrix_array_2 = Array2::from_shape_vec(
        (num_rows, num_cols),
        matrix.mle.bookkeeping_table().to_vec(),
    )
    .unwrap();
    let matrix_transpose = matrix_array_2.reversed_axes();
    let matrix_transp_vec = matrix_transpose
        .outer_iter()
        .map(|x| x.to_vec())
        .flat_map(|row| row)
        .collect_vec();

    let mle = DenseMle::new_with_indices(
        &matrix_transp_vec,
        matrix.mle.layer_id,
        &matrix.mle.mle_indices,
    );

    Matrix::new(mle, num_rows, num_cols)
}

/// Multiply two matrices together, with a transposed matrix_b
pub fn product_two_matrices<F: FieldExt>(matrix_a: &Matrix<F>, matrix_b: &Matrix<F>) -> Vec<F> {
    let num_middle_ab = 1 << matrix_a.num_cols_vars;

    let matrix_b_transpose = gen_transpose_matrix(matrix_b);

    let product_matrix = matrix_a
        .mle
        .bookkeeping_table()
        .chunks(num_middle_ab as usize)
        .flat_map(|chunk_a| {
            matrix_b_transpose
                .mle
                .bookkeeping_table()
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
    use remainder_shared_types::Fr;

    use crate::{
        claims::Claim,
        layer::{
            matmult::{product_two_matrices, MatMult, Matrix},
            LayerId,
        },
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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 4, 2);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 2, 2);

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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 8, 4);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 4, 2);

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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 5, 3);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 3, 3);

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

        let matrix_out = Matrix::new(DenseMle::new_from_raw(exp_product, LayerId::Layer(0)), 5, 3);

        assert_eq!(res_product, matrix_out.mle.bookkeeping_table().clone());
    }

    #[test]
    /// super basic symmetric test
    fn test_sumcheck_1() {
        let mut rng = test_rng();
        let claim = Claim::new(vec![Fr::from(1), Fr::from(0)], Fr::from(3));

        let matrix_a_vec = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let matrix_b_vec = vec![Fr::from(1), Fr::from(1), Fr::from(1), Fr::from(1)];

        let matrix_a = Matrix::new(
            DenseMle::new_from_raw(matrix_a_vec, LayerId::Layer(0)),
            2,
            2,
        );
        let matrix_b = Matrix::new(
            DenseMle::new_from_raw(matrix_b_vec, LayerId::Layer(0)),
            2,
            2,
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

        let matrix_a = Matrix::new(
            DenseMle::new_from_raw(matrix_a_vec, LayerId::Layer(0)),
            2,
            4,
        );
        let matrix_b = Matrix::new(
            DenseMle::new_from_raw(matrix_b_vec, LayerId::Layer(0)),
            4,
            1,
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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 5, 3);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 3, 3);

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

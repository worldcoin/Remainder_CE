//! This module contains the implementation of the matrix multiplication layer

use std::collections::HashSet;

use ::serde::{Deserialize, Serialize};
use ark_std::cfg_into_iter;
use itertools::Itertools;
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};

use super::{
    combine_mle_refs::{combine_mle_refs_with_aggregate, pre_fix_mle_refs},
    gate::compute_sumcheck_message_no_beta_table,
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    regular_layer::claims::CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION,
    CircuitLayer, Layer, LayerError, LayerId, VerifierLayer,
};
use crate::{
    claims::{
        wlx_eval::{get_num_wlx_evaluations, ClaimMle, YieldWLXEvals},
        Claim, ClaimError, YieldClaim,
    },
    expression::{circuit_expr::CircuitMle, verifier_expr::VerifierMle},
    layer::VerificationError,
    layouter::layouting::{CircuitLocation, CircuitMap},
    mle::{dense::DenseMle, evals::MultilinearExtension, Mle, MleIndex},
    sumcheck::evaluate_at_a_point,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Used to represent a matrix, along with its optional prefix bits (in circuit)
/// basically an equivalence of DenseMle<F>, but uninstantiated, until preprocessing
/// TODO(vishady): NEED TO PAD THIS BEFORE CALLING `::new` SO IT SHOULD ALWAYS TAKE IN LOG DIMENSIONS!
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct Matrix<F: Field> {
    /// The underlying and padded MLE that represents this matrix.
    pub mle: DenseMle<F>,
    rows_num_vars: usize,
    cols_num_vars: usize,
}

impl<F: Field> Matrix<F> {
    /// Create a new matrix.
    pub fn new(mle: DenseMle<F>, rows_num_vars: usize, cols_num_vars: usize) -> Matrix<F> {
        assert_eq!(
            mle.bookkeeping_table().len(),
            (1 << rows_num_vars) * (1 << cols_num_vars)
        );

        Matrix {
            mle,
            rows_num_vars,
            cols_num_vars,
        }
    }

    /// get the dimension of this matrix
    pub fn rows_cols_num_vars(&self) -> (usize, usize) {
        (self.rows_num_vars, self.cols_num_vars)
    }
}

/// Used to represent a matrix multiplication layer
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "F: Field")]
pub struct MatMult<F: Field> {
    layer_id: LayerId,
    matrix_a: Matrix<F>,
    matrix_b: Matrix<F>,
    num_vars_middle_ab: Option<usize>,
}

impl<F: Field> MatMult<F> {
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
            (1 << self.matrix_a.cols_num_vars) * (1 << self.matrix_a.rows_num_vars),
            matrix_a_mle.bookkeeping_table().len()
        );
        assert_eq!(
            (1 << self.matrix_b.cols_num_vars) * (1 << self.matrix_b.rows_num_vars),
            matrix_b_mle.bookkeeping_table().len()
        );

        // check to make sure the dimensions match
        if self.matrix_a.cols_num_vars == self.matrix_b.rows_num_vars {
            self.num_vars_middle_ab = Some(self.matrix_a.cols_num_vars);
        } else {
            panic!("Matrix dimensions do not match")
        }

        let matrix_a_transp = gen_transpose_matrix(&self.matrix_a);

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
            .chain(bound_indices_a)
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

    fn append_leaf_mles_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<F>) {
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

impl<F: Field> Layer<F> for MatMult<F> {
    fn prove_rounds(
        &mut self,
        claim: Claim<F>,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) -> Result<(), LayerError> {
        let mut claim_b = claim.get_point().clone();
        let claim_a = claim_b.split_off(self.matrix_b.cols_num_vars);
        self.pre_processing_step(claim_a, claim_b);

        let mut challenges: Vec<F> = vec![];
        let num_vars_middle = self.num_vars_middle_ab.unwrap(); // TODO: raise error if not

        for round in 0..num_vars_middle {
            let message = compute_sumcheck_message_no_beta_table(
                &[&self.matrix_a.mle, &self.matrix_b.mle],
                round,
                2,
            )
            .unwrap();
            transcript_writer.append_elements("Sumcheck evaluations", &message);
            let challenge = transcript_writer.get_challenge("Sumcheck challenge");
            challenges.push(challenge);
            self.matrix_a.mle.fix_variable(round, challenge);
            self.matrix_b.mle.fix_variable(round, challenge);
        }
        self.append_leaf_mles_to_transcript(transcript_writer);
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn initialize_sumcheck(&mut self, claim_point: &[F]) -> Result<(), LayerError> {
        let mut claim_b = claim_point.to_vec();
        let claim_a = claim_b.split_off(self.matrix_b.cols_num_vars);
        self.pre_processing_step(claim_a, claim_b);
        Ok(())
    }

    fn compute_round_sumcheck_message(&self, round_index: usize) -> Result<Vec<F>, LayerError> {
        let mles = vec![&self.matrix_a.mle, &self.matrix_b.mle];
        let sumcheck_message =
            compute_sumcheck_message_no_beta_table(&mles, round_index, 2).unwrap();
        Ok(sumcheck_message)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: F) -> Result<(), LayerError> {
        self.matrix_a.mle.fix_variable(round_index, challenge);
        self.matrix_b.mle.fix_variable(round_index, challenge);

        Ok(())
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.num_vars_middle_ab.unwrap()).collect_vec()
    }

    fn max_degree(&self) -> usize {
        2
    }

    /// Return the [PostSumcheckLayer], panicking if either of the MLE refs is not fully bound.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        _round_challenges: &[F],
        _claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, F> {
        let mle_refs = vec![self.matrix_a.mle.clone(), self.matrix_b.mle.clone()];
        PostSumcheckLayer(vec![Product::<F, F>::new(&mle_refs, F::ONE)])
    }
}

/// The circuit description counterpart of a [Matrix].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct CircuitMatrix<F: Field> {
    mle: CircuitMle<F>,
    rows_num_vars: usize,
    cols_num_vars: usize,
}

impl<F: Field> CircuitMatrix<F> {
    /// The constructor for a [CircuitMatrix], which is the circuit
    /// description of matrix, only containing shape information
    /// which is the number of variables in the rows and the number
    /// of variables in the columns.
    pub fn new(mle: CircuitMle<F>, rows_num_vars: usize, cols_num_vars: usize) -> Self {
        Self {
            mle,
            rows_num_vars,
            cols_num_vars,
        }
    }

    /// Convert the circuit description of a matrix into the prover
    /// view of a matrix, using the [CircuitMap].
    pub fn into_matrix(&self, circuit_map: &CircuitMap<F>) -> Matrix<F> {
        let dense_mle = self.mle.into_dense_mle(circuit_map);
        Matrix {
            mle: dense_mle,
            rows_num_vars: self.rows_num_vars,
            cols_num_vars: self.cols_num_vars,
        }
    }
}
/// The circuit description counterpart of a [MatMult] layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct CircuitMatMultLayer<F: Field> {
    /// The layer id associated with this matmult layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: CircuitMatrix<F>,

    /// The RHS Matrix to be multiplied.
    matrix_b: CircuitMatrix<F>,
}

impl<F: Field> CircuitMatMultLayer<F> {
    /// Constructor for the [CircuitMatMultLayer], using the circuit description
    /// of the matrices that make up this layer.
    pub fn new(layer_id: LayerId, matrix_a: CircuitMatrix<F>, matrix_b: CircuitMatrix<F>) -> Self {
        Self {
            layer_id,
            matrix_a,
            matrix_b,
        }
    }
}

impl<F: Field> CircuitLayer<F> for CircuitMatMultLayer<F> {
    type VerifierLayer = VerifierMatMultLayer<F>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn verify_rounds(
        &self,
        claim: Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<VerifierLayerEnum<F>, VerificationError> {
        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
        let mut g_prev_round = vec![claim.get_result()];

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = F::ZERO;

        // Get the number of rounds, which is exactly the inner dimension of the matrix product.
        assert_eq!(self.matrix_a.cols_num_vars, self.matrix_b.rows_num_vars);
        let num_rounds = self.matrix_a.cols_num_vars;

        // For round 1 <= i <= n, perform the check:
        for _round in 0..num_rounds {
            let degree = 2;

            let g_cur_round = transcript_reader
                .consume_elements("Sumcheck message", degree + 1)
                .map_err(VerificationError::TranscriptError)?;

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

        let verifier_layer: VerifierMatMultLayer<F> = self
            .convert_into_verifier_layer(&challenges, claim.get_point(), transcript_reader)
            .unwrap();

        let matrix_product = verifier_layer.evaluate();

        if g_final_r_final != matrix_product {
            return Err(VerificationError::FinalSumcheckFailed);
        }

        Ok(VerifierLayerEnum::MatMult(verifier_layer))
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        assert_eq!(self.matrix_a.cols_num_vars, self.matrix_b.rows_num_vars);
        (0..self.matrix_a.cols_num_vars).collect_vec()
    }

    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&CircuitMle<F>>,
        circuit_map: &mut CircuitMap<F>,
    ) -> bool {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();

        let maybe_matrix_a_data = circuit_map.get_data_from_circuit_mle(&self.matrix_a.mle);
        if maybe_matrix_a_data.is_err() {
            return false;
        }
        let matrix_a_data = maybe_matrix_a_data.unwrap();
        let maybe_matrix_b_data = circuit_map.get_data_from_circuit_mle(&self.matrix_b.mle);
        if maybe_matrix_b_data.is_err() {
            return false;
        }
        let matrix_b_data = maybe_matrix_b_data.unwrap();
        let product = product_two_matrices_from_flattened_vectors(
            matrix_a_data.get_evals_vector(),
            matrix_b_data.get_evals_vector(),
            1 << self.matrix_a.rows_num_vars,
            1 << self.matrix_a.cols_num_vars,
            1 << self.matrix_b.rows_num_vars,
            1 << self.matrix_b.cols_num_vars,
        );

        let output_data = MultilinearExtension::new(product);
        assert_eq!(
            output_data.num_vars(),
            mle_output_necessary.mle_indices().len()
        );

        circuit_map.add_node(CircuitLocation::new(self.layer_id(), vec![]), output_data);
        true
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[F],
        claim_point: &[F],
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::VerifierLayer, VerificationError> {
        // Split the claim into the claims made on matrix A rows and matrix B cols.
        let mut claim_b = claim_point.to_vec();
        let claim_a = claim_b.split_off(self.matrix_b.cols_num_vars);

        // Construct the full claim made on A using the claim made on the layer and the sumcheck bindings.
        let full_claim_chals_a = sumcheck_bindings
            .iter()
            .copied()
            .chain(claim_a)
            .collect_vec();

        // Construct the full claim made on B using the claim made on the layer and the sumcheck bindings.
        let full_claim_chals_b = claim_b
            .into_iter()
            .chain(sumcheck_bindings.to_vec())
            .collect_vec();

        // Shape checks.
        assert_eq!(
            full_claim_chals_a.len(),
            self.matrix_a.rows_num_vars + self.matrix_a.cols_num_vars
        );
        assert_eq!(
            full_claim_chals_b.len(),
            self.matrix_b.rows_num_vars + self.matrix_b.cols_num_vars
        );

        // Construct the verifier matrices given these fully bound points.
        let matrix_a = VerifierMatrix {
            mle: self
                .matrix_a
                .mle
                .into_verifier_mle(&full_claim_chals_a, transcript_reader)
                .unwrap(),
            rows_num_vars: self.matrix_a.rows_num_vars,
            cols_num_vars: self.matrix_a.cols_num_vars,
        };
        let matrix_b = VerifierMatrix {
            mle: self
                .matrix_b
                .mle
                .into_verifier_mle(&full_claim_chals_b, transcript_reader)
                .unwrap(),
            rows_num_vars: self.matrix_b.rows_num_vars,
            cols_num_vars: self.matrix_b.cols_num_vars,
        };

        Ok(VerifierMatMultLayer {
            layer_id: self.layer_id,
            matrix_a,
            matrix_b,
        })
    }

    /// Return the PostSumcheckLayer, given challenges that fully bind the expression.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[F],
        claim_challenges: &[F],
    ) -> PostSumcheckLayer<F, Option<F>> {
        let mut pre_bound_matrix_a_mle = self.matrix_a.mle.clone();
        let claim_chals_matrix_a = claim_challenges[self.matrix_b.cols_num_vars..].to_vec();
        let mut indexed_index_counter = 0;
        let mut bound_index_counter = 0;

        let matrix_a_new_indices = self
            .matrix_a
            .mle
            .mle_indices()
            .iter()
            .map(|mle_idx| match mle_idx {
                &MleIndex::IndexedBit(_) => {
                    if indexed_index_counter < self.matrix_a.cols_num_vars {
                        let ret = MleIndex::IndexedBit(indexed_index_counter);
                        indexed_index_counter += 1;
                        ret
                    } else {
                        let ret = MleIndex::Bound(
                            claim_chals_matrix_a[bound_index_counter],
                            bound_index_counter,
                        );
                        bound_index_counter += 1;
                        ret
                    }
                }
                MleIndex::Fixed(_) => mle_idx.clone(),
                MleIndex::Iterated => panic!("should not have any iterated indices"),
                MleIndex::Bound(_, _) => panic!("should not have any bound indices"),
            })
            .collect_vec();
        pre_bound_matrix_a_mle.set_mle_indices(matrix_a_new_indices);

        let mut pre_bound_matrix_b_mle = self.matrix_b.mle.clone();
        let claim_chals_matrix_b = claim_challenges[..self.matrix_b.cols_num_vars].to_vec();
        let mut bound_index_counter = 0;
        let mut indexed_index_counter = 0;
        let matrix_b_new_indices = self
            .matrix_b
            .mle
            .mle_indices()
            .iter()
            .map(|mle_idx| match mle_idx {
                &MleIndex::IndexedBit(_) => {
                    if bound_index_counter < self.matrix_b.cols_num_vars {
                        let ret = MleIndex::Bound(
                            claim_chals_matrix_b[bound_index_counter],
                            bound_index_counter,
                        );
                        bound_index_counter += 1;
                        ret
                    } else {
                        let ret = MleIndex::IndexedBit(indexed_index_counter);
                        indexed_index_counter += 1;
                        ret
                    }
                }
                MleIndex::Fixed(_) => mle_idx.clone(),
                MleIndex::Iterated => panic!("should not have any iterated indices"),
                MleIndex::Bound(_, _) => panic!("should not have any bound indices"),
            })
            .collect_vec();
        pre_bound_matrix_b_mle.set_mle_indices(matrix_b_new_indices);
        let mle_refs = vec![pre_bound_matrix_a_mle, pre_bound_matrix_b_mle];

        PostSumcheckLayer(vec![Product::<F, Option<F>>::new(
            &mle_refs,
            F::ONE,
            round_challenges,
        )])
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_circuit_mles(&self) -> Vec<&CircuitMle<F>> {
        vec![&self.matrix_a.mle, &self.matrix_b.mle]
    }

    fn convert_into_prover_layer<'a>(&self, circuit_map: &CircuitMap<F>) -> LayerEnum<F> {
        let prover_matrix_a = self.matrix_a.into_matrix(circuit_map);
        let prover_matrix_b = self.matrix_b.into_matrix(circuit_map);
        let matmult_layer = MatMult::new(self.layer_id, prover_matrix_a, prover_matrix_b);
        matmult_layer.into()
    }

    fn index_mle_indices(&mut self, start_index: usize) {
        self.matrix_a.mle.index_mle_indices(start_index);
        self.matrix_b.mle.index_mle_indices(start_index);
    }
}

/// The verifier's counterpart of a [Matrix].
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct VerifierMatrix<F: Field> {
    mle: VerifierMle<F>,
    rows_num_vars: usize,
    cols_num_vars: usize,
}

/// The verifier's counterpart of a [MatMult] layer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound = "F: Field")]
pub struct VerifierMatMultLayer<F: Field> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: VerifierMatrix<F>,

    /// The RHS Matrix to be multiplied.
    matrix_b: VerifierMatrix<F>,
}

impl<F: Field> VerifierLayer<F> for VerifierMatMultLayer<F> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}

impl<F: Field> VerifierMatMultLayer<F> {
    fn evaluate(&self) -> F {
        self.matrix_a.mle.value() * self.matrix_b.mle.value()
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for VerifierMatMultLayer<F> {
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let claims = vec![&self.matrix_a, &self.matrix_b]
            .into_iter()
            .map(|matrix| {
                let matrix_fixed_indices = matrix
                    .mle
                    .mle_indices()
                    .iter()
                    .map(|index| {
                        index
                            .val()
                            .ok_or(LayerError::ClaimError(ClaimError::ClaimMleIndexError))
                            .unwrap()
                    })
                    .collect_vec();

                let matrix_claimed_val = matrix.mle.value();

                let claim: ClaimMle<F> = ClaimMle::new(
                    matrix_fixed_indices,
                    matrix_claimed_val,
                    Some(self.layer_id),
                    Some(matrix.mle.layer_id()),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

impl<F: Field> YieldClaim<ClaimMle<F>> for MatMult<F> {
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<ClaimMle<F>>, LayerError> {
        let claims = vec![&self.matrix_a.mle, &self.matrix_b.mle]
            .into_iter()
            .map(|matrix_mle| {
                let matrix_fixed_indices = matrix_mle
                    .mle_indices()
                    .iter()
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
                    Some(self.layer_id),
                    Some(matrix_mle.layer_id),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

impl<F: Field> YieldWLXEvals<F> for MatMult<F> {
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claim_mles: Vec<DenseMle<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, ClaimError> {
        // get the number of evaluations
        let (num_evals, common_idx) = if CLAIM_AGGREGATION_CONSTANT_COLUMN_OPTIMIZATION {
            let (num_evals, common_idx, _) = get_num_wlx_evaluations(claim_vecs);
            (num_evals, common_idx)
        } else {
            (((num_claims - 1) * num_idx) + 1, None)
        };

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

impl<F: std::fmt::Debug + Field> MatMult<F> {
    pub(crate) fn circuit_description_fmt(&self) -> impl std::fmt::Display + '_ {
        // Dummy struct which simply exists to implement `std::fmt::Display`
        // so that it can be returned as an `impl std::fmt::Display`
        struct MatMultCircuitDesc<'a, F: std::fmt::Debug + Field>(&'a MatMult<F>);

        impl<'a, F: std::fmt::Debug + Field> std::fmt::Display for MatMultCircuitDesc<'a, F> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("MatMult")
                    .field("matrix_a_layer_id", &self.0.matrix_a.mle.layer_id)
                    .field("matrix_a_mle_indices", &self.0.matrix_a.mle.mle_indices)
                    .field("matrix_b_layer_id", &self.0.matrix_b.mle.layer_id)
                    .field("matrix_b_mle_indices", &self.0.matrix_b.mle.mle_indices)
                    .field("num_vars_middle_ab", &self.0.matrix_a.cols_num_vars)
                    .finish()
            }
        }
        MatMultCircuitDesc(self)
    }
}

/// Generate the transpose of a matrix, uses Array2 from ndarray
pub fn gen_transpose_matrix<F: Field>(matrix: &Matrix<F>) -> Matrix<F> {
    let num_rows = 1 << matrix.rows_num_vars;
    let num_cols = 1 << matrix.cols_num_vars;

    // Memory-efficient, sequential implementation.
    let mut matrix_transp_vec = Vec::with_capacity(num_cols * num_rows);

    matrix.mle.bookkeeping_table();
    for i in 0..num_cols {
        for j in 0..num_rows {
            matrix_transp_vec.push(matrix.mle.mle[j * num_cols + i]);
        }
    }

    let mle = DenseMle::new_with_indices(
        &matrix_transp_vec,
        matrix.mle.layer_id,
        &matrix.mle.mle_indices,
    );

    Matrix::new(mle, matrix.cols_num_vars, matrix.rows_num_vars)
}

/// Multiply two matrices together, with a transposed matrix_b
pub fn product_two_matrices<F: Field>(matrix_a: &Matrix<F>, matrix_b: &Matrix<F>) -> Vec<F> {
    let num_middle_ab = 1 << matrix_a.cols_num_vars;

    let matrix_b_transpose = gen_transpose_matrix(matrix_b);

    let product_matrix = matrix_a
        .mle
        .bookkeeping_table()
        .chunks(num_middle_ab)
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

/// Compute the product of two matrices given flattened vectors rather than
/// matrices.
pub fn product_two_matrices_from_flattened_vectors<F: Field>(
    matrix_a_vec: &[F],
    matrix_b_vec: &[F],
    matrix_a_num_rows: usize,
    matrix_a_num_cols: usize,
    matrix_b_num_rows: usize,
    matrix_b_num_cols: usize,
) -> Vec<F> {
    assert_eq!(
        matrix_a_num_cols, matrix_b_num_rows,
        "Matrix dimensions are not compatible for multiplication"
    );

    let mut result = vec![F::ZERO; matrix_a_num_rows * matrix_b_num_cols];

    for i in 0..matrix_a_num_rows {
        for j in 0..matrix_b_num_cols {
            for k in 0..matrix_a_num_cols {
                result[i * matrix_b_num_cols + j] += matrix_a_vec[i * matrix_a_num_cols + k]
                    * matrix_b_vec[k * matrix_b_num_cols + j];
            }
        }
    }

    result
}

#[cfg(test)]
mod test {

    use remainder_shared_types::Fr;

    use crate::{
        layer::{
            matmult::{product_two_matrices, Matrix},
            LayerId,
        },
        mle::dense::DenseMle,
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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 2, 1);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 1, 1);

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

        let matrix_a = Matrix::new(DenseMle::new_from_raw(mle_vec_a, LayerId::Layer(0)), 3, 2);
        let matrix_b = Matrix::new(DenseMle::new_from_raw(mle_vec_b, LayerId::Layer(0)), 2, 1);

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
}

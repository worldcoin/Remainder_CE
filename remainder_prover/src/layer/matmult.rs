//! This module contains the implementation of the matrix multiplication layer

use std::collections::HashSet;

use ::serde::{Deserialize, Serialize};
use itertools::Itertools;
use remainder_shared_types::{
    extension_field::ExtensionField, transcript::{ProverTranscript, VerifierTranscript}, Field
};

use super::{
    gate::compute_sumcheck_message_no_beta_table,
    layer_enum::{LayerEnum, VerifierLayerEnum},
    product::{PostSumcheckLayer, Product},
    Layer, LayerDescription, LayerError, LayerId, VerifierLayer,
};
use crate::{
    circuit_layout::{CircuitEvalMap, CircuitLocation},
    claims::{Claim, ClaimError, RawClaim},
    layer::VerificationError,
    mle::{
        dense::DenseMle, evals::MultilinearExtension, mle_description::MleDescription, verifier_mle::VerifierMle, AbstractMle, Mle, MleIndex
    },
    sumcheck::evaluate_at_a_point,
};

use anyhow::{anyhow, Ok, Result};

/// Used to represent a matrix; basically an MLE which is the
/// flattened version of this matrix along with the log2
/// num_rows (`rows_num_vars`) and the log2 num_cols `cols_num_vars`.
///
/// This ensures that the flattened MLE provided already has
/// a bookkeeping table where the rows and columns are padded to
/// the nearest power of 2.
///
/// NOTE: the flattened MLE that represents a matrix is in
/// row major order. Internal bookkeeping tables are
/// stored in big-endian, so the FIRST "row"
/// number of variables to represent the rows of the matrix,
/// and the LAST "column" number of variables to represent the
/// columns of the matrix.
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
        assert_eq!(mle.len(), (1 << rows_num_vars) * (1 << cols_num_vars));

        Matrix {
            mle,
            rows_num_vars,
            cols_num_vars,
        }
    }

    /// Get the dimensions of this matrix.
    pub fn rows_cols_num_vars(&self) -> (usize, usize) {
        (self.rows_num_vars, self.cols_num_vars)
    }
}

/// Used to represent a matrix multiplication layer.
///
/// #Attributes:
/// * `layer_id` - the LayerId of this MatMult layer.
/// * `matrix_a` - the lefthand side matrix in the multiplication.
/// * `matrix_b` - the righthand side matrix in the multiplication.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound = "E: ExtensionField")]
pub struct MatMult<E: ExtensionField> {
    layer_id: LayerId,
    matrix_a: Matrix<E>,
    matrix_b: Matrix<E>,
    num_vars_middle_ab: usize,
}

impl<E: ExtensionField> MatMult<E> {
    /// Create a new matrix multiplication layer.
    pub fn new(layer_id: LayerId, matrix_a: Matrix<E>, matrix_b: Matrix<E>) -> MatMult<E> {
        // Check to make sure the inner dimensions of the matrices we are
        // producting match. I.e., the number of variables representing the
        // columns of matrix a are the same as the number of variables
        // representing the rows of matrix b.
        assert_eq!(matrix_a.cols_num_vars, matrix_b.rows_num_vars);
        let num_vars_middle_ab = matrix_a.cols_num_vars;
        MatMult {
            layer_id,
            matrix_a: matrix_a,
            matrix_b: matrix_b,
            num_vars_middle_ab,
        }
    }

    /// The step, according to [Tha13](https://eprint.iacr.org/2013/351.pdf), which
    /// makes the matmult algorithm super-efficient.
    ///
    /// Given the claim on the output of the matrix multiplication, we bind
    /// the variables representing the "rows" of `matrix_a` to the first
    /// log(num_rows_a) vars in this claim, and we bind the variables
    /// representing the "columns" of `matrix_b` to the last log(num_cols_b)
    /// vars in the claim.
    ///
    /// #Arguments
    /// * `claim_a`: the first log_num_rows variables of the claim made on the
    ///   MLE representing the output of this layer.
    /// * `claim_b`: the last log_num_cols variables of the claim made on the
    ///   MLE representing the output of this layer.
    fn pre_processing_step(&mut self, claim_a: Vec<E>, claim_b: Vec<E>) {
        let matrix_a_mle = &mut self.matrix_a.mle;
        let matrix_b_mle = &mut self.matrix_b.mle;

        // Check that both matrices are padded such that the number of rows
        // and the number of columns are both powers of 2.
        assert_eq!(
            (1 << self.matrix_a.cols_num_vars) * (1 << self.matrix_a.rows_num_vars),
            matrix_a_mle.len()
        );
        assert_eq!(
            (1 << self.matrix_b.cols_num_vars) * (1 << self.matrix_b.rows_num_vars),
            matrix_b_mle.len()
        );

        matrix_a_mle.index_mle_indices(0);
        matrix_b_mle.index_mle_indices(0);

        // Bind the row indices of matrix A to the relevant claim point.
        claim_a.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_a_mle.fix_variable(idx, chal);
        });

        // Bind the column indices of matrix B to the relevant claim point.
        claim_b.into_iter().enumerate().for_each(|(idx, chal)| {
            matrix_b_mle.fix_variable_at_index(idx + self.matrix_b.rows_num_vars, chal);
        });
        // We want to re-index the MLE indices in matrix A such that it
        // starts from 0 after the pre-processing, so we do that by first
        // setting them to be free and then re-indexing them.
        let new_a_indices = matrix_a_mle
            .clone()
            .mle_indices
            .into_iter()
            .map(|index| {
                if let MleIndex::Indexed(_) = index {
                    MleIndex::Free
                } else {
                    index
                }
            })
            .collect_vec();
        matrix_a_mle.mle_indices = new_a_indices;
        matrix_a_mle.index_mle_indices(0);
    }

    fn append_leaf_mles_to_transcript(&self, transcript_writer: &mut impl ProverTranscript<E::BaseField>) {
        transcript_writer.append_extension_field_elements(
            "Fully bound MLE evaluation",
            &[self.matrix_a.mle.value(), self.matrix_b.mle.value()],
        );
    }
}

impl<E: ExtensionField> Layer<E> for MatMult<E> {
    // Since we pre-process the matrices first, by pre-binding the
    // row variables of matrix A and the column variables of matrix B,
    // the number of rounds of sumcheck is simply the number of variables
    // that represent the singular (same) inner dimension of both of the
    // matrices in this matrix product.
    fn prove(
        &mut self,
        claims: &[&RawClaim<E>],
        transcript_writer: &mut impl ProverTranscript<E::BaseField>,
    ) -> Result<()> {
        println!(
            "MatMul::prove_rounds() for a product ({} x {}) * ({} x {}) matrix.",
            self.matrix_a.rows_num_vars,
            self.matrix_a.cols_num_vars,
            self.matrix_b.rows_num_vars,
            self.matrix_b.cols_num_vars
        );

        // We always use interpolative claim aggregation for matmult layers
        // because the preprocessing step in matmult utilizes the fact that we
        // have linear variables in the expression, which RLC is unable to
        // aggregate claims for.
        assert_eq!(claims.len(), 1);
        self.initialize(claims[0].get_point())?;

        let num_vars_middle = self.num_vars_middle_ab;

        for round in 0..num_vars_middle {
            // Compute the round's sumcheck message.
            let message = self.compute_round_sumcheck_message(round, &[E::ONE])?;
            // Add to transcript.
            // Since the verifier can deduce g_i(0) by computing claim - g_i(1), the prover does not send g_i(0)
            transcript_writer
                .append_extension_field_elements("Sumcheck round univariate evaluations", &message[1..]);
            // Sample the challenge to bind the round's MatMult expression to.
            let challenge = transcript_writer.get_extension_field_challenge("Sumcheck round challenge");
            // Bind the Matrix MLEs to this variable.
            self.bind_round_variable(round, challenge)?;
        }

        // Assert that the MLEs have been fully bound.
        assert!(self.matrix_a.mle.is_fully_bounded());
        assert!(self.matrix_b.mle.is_fully_bounded());

        self.append_leaf_mles_to_transcript(transcript_writer);
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn initialize(&mut self, claim_point: &[E]) -> Result<()> {
        // Split the claim on the MLE representing the output of this layer
        // accordingly.
        // We need to make sure the number of variables in the claim is the
        // sum of the outer dimensions of this matrix product.
        assert_eq!(
            claim_point.len(),
            self.matrix_a.rows_num_vars + self.matrix_b.cols_num_vars
        );
        let mut claim_a = claim_point.to_vec();
        let claim_b = claim_a.split_off(self.matrix_a.rows_num_vars);
        self.pre_processing_step(claim_a, claim_b);
        Ok(())
    }

    fn initialize_rlc(&mut self, _random_coefficients: &[E], _claims: &[&RawClaim<E>]) {
        // This function is not implemented for MatMult layers because we should
        // never be using RLC claim aggregation for MatMult layers. Instead, we always
        // use interpolative claim aggregation.
        unimplemented!()
    }

    fn compute_round_sumcheck_message(
        &mut self,
        round_index: usize,
        _random_coefficients: &[E],
    ) -> Result<Vec<E>> {
        let mles = vec![&self.matrix_a.mle, &self.matrix_b.mle];
        let sumcheck_message =
            compute_sumcheck_message_no_beta_table(&mles, round_index, 2).unwrap();
        Ok(sumcheck_message)
    }

    fn bind_round_variable(&mut self, round_index: usize, challenge: E) -> Result<()> {
        self.matrix_a.mle.fix_variable(round_index, challenge);
        self.matrix_b.mle.fix_variable(round_index, challenge);

        Ok(())
    }

    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.num_vars_middle_ab).collect_vec()
    }

    fn max_degree(&self) -> usize {
        2
    }

    /// Return the [PostSumcheckLayer], panicking if either of the MLE refs is not fully bound.
    /// Relevant for the Hyrax IP, where we need commitments to fully bound MLEs as well as their intermediate products.
    fn get_post_sumcheck_layer(
        &self,
        _round_challenges: &[E],
        _claim_challenges: &[&[E]],
        _random_coefficients: &[E],
    ) -> PostSumcheckLayer<E, E> {
        let mles = vec![self.matrix_a.mle.clone(), self.matrix_b.mle.clone()];
        PostSumcheckLayer(vec![Product::<E, E>::new(&mles, E::ONE)])
    }
    /// Get the claims that this layer makes on other layers
    fn get_claims(&self) -> Result<Vec<Claim<E>>> {
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

                let matrix_val = matrix_mle.value();
                let claim: Claim<E> = Claim::new(
                    matrix_fixed_indices,
                    matrix_val,
                    self.layer_id,
                    matrix_mle.layer_id,
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

/// The circuit description counterpart of a [Matrix].
#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
#[serde(bound = "F: Field")]
pub struct MatrixDescription<F: Field> {
    mle: MleDescription<F>,
    rows_num_vars: usize,
    cols_num_vars: usize,
}

impl<F: Field> MatrixDescription<F> {
    /// The constructor for a [MatrixDescription], which is the circuit
    /// description of matrix, only containing shape information
    /// which is the number of variables in the rows and the number
    /// of variables in the columns.
    pub fn new(mle: MleDescription<F>, rows_num_vars: usize, cols_num_vars: usize) -> Self {
        Self {
            mle,
            rows_num_vars,
            cols_num_vars,
        }
    }

    /// Convert the circuit description of a matrix into the prover
    /// view of a matrix, using the [CircuitMap].
    pub fn into_matrix(&self, circuit_map: &CircuitEvalMap<F>) -> Matrix<F> {
        let dense_mle = self.mle.into_dense_mle(circuit_map);
        Matrix {
            mle: dense_mle,
            rows_num_vars: self.rows_num_vars,
            cols_num_vars: self.cols_num_vars,
        }
    }
}
/// The circuit description counterpart of a [MatMult] layer.
#[derive(Serialize, Deserialize, Clone, Debug, Hash)]
#[serde(bound = "F: Field")]
pub struct MatMultLayerDescription<F: Field> {
    /// The layer id associated with this matmult layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: MatrixDescription<F>,

    /// The RHS Matrix to be multiplied.
    matrix_b: MatrixDescription<F>,
}

impl<F: Field> MatMultLayerDescription<F> {
    /// Constructor for the [MatMultLayerDescription], using the circuit description
    /// of the matrices that make up this layer.
    pub fn new(
        layer_id: LayerId,
        matrix_a: MatrixDescription<F>,
        matrix_b: MatrixDescription<F>,
    ) -> Self {
        Self {
            layer_id,
            matrix_a,
            matrix_b,
        }
    }
}

impl<E: ExtensionField> LayerDescription<E> for MatMultLayerDescription<E> {
    type VerifierLayer = VerifierMatMultLayer<E>;

    /// Gets this layer's id.
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn verify_rounds(
        &self,
        claims: &[&RawClaim<E>],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<VerifierLayerEnum<E>> {
        // Keeps track of challenges `r_1, ..., r_n` sent by the verifier.
        let mut challenges = vec![];

        // For matmult we always use the interpolative claim aggregation method.
        assert_eq!(claims.len(), 1);
        let claim = claims[0];

        // Represents `g_{i-1}(x)` of the previous round.
        // This is initialized to the constant polynomial `g_0(x)` which evaluates
        // to the claim result for any `x`.
        let mut g_prev_round = vec![claim.get_eval()];

        // Previous round's challege: r_{i-1}.
        let mut prev_challenge = E::ZERO;

        // Get the number of rounds, which is exactly the inner dimension of the matrix product.
        assert_eq!(self.matrix_a.cols_num_vars, self.matrix_b.rows_num_vars);
        let num_rounds = self.matrix_a.cols_num_vars;

        // For round 1 <= i <= n, perform the check:
        for _round in 0..num_rounds {
            let degree = 2;

            // Read g_i(1), ..., g_i(d+1) from the prover, reserve space to compute g_i(0)
            let mut g_cur_round: Vec<_> = [Ok(E::from(0))]
                .into_iter()
                .chain((0..degree).map(|_| {
                    transcript_reader.consume_extension_field_element("Sumcheck round univariate evaluations")
                }))
                .collect::<Result<_, _>>()?;

            // Sample random challenge `r_i`.
            let challenge = transcript_reader.get_extension_field_challenge("Sumcheck round challenge")?;

            // Compute:
            //       `g_i(0) = g_{i - 1}(r_{i-1}) - g_i(1)`
            let g_prev_r_prev = evaluate_at_a_point(&g_prev_round, prev_challenge).unwrap();
            let g_i_one = evaluate_at_a_point(&g_cur_round, E::ONE).unwrap();
            g_cur_round[0] = g_prev_r_prev - g_i_one;

            g_prev_round = g_cur_round;
            prev_challenge = challenge;
            challenges.push(challenge);
        }

        // Evalute `g_n(r_n)`.
        // Note: If there were no nonlinear rounds, this value reduces to
        // `claim.get_result()` due to how we initialized `g_prev_round`.
        let g_final_r_final = evaluate_at_a_point(&g_prev_round, prev_challenge)?;

        let verifier_layer: VerifierMatMultLayer<E> = self
            .convert_into_verifier_layer(&challenges, &[claim.get_point()], transcript_reader)
            .unwrap();

        let matrix_product = verifier_layer.evaluate();

        if g_final_r_final != matrix_product {
            return Err(anyhow!(VerificationError::FinalSumcheckFailed));
        }

        Ok(VerifierLayerEnum::MatMult(verifier_layer))
    }

    /// The number of sumcheck rounds are only those over the inner dimensions
    /// of the matrix, hence they enumerate from 0 to the inner dimension.
    fn sumcheck_round_indices(&self) -> Vec<usize> {
        (0..self.matrix_a.cols_num_vars).collect_vec()
    }

    /// Compute the evaluations of the MLE that represents the
    /// product of the two matrices over the boolean hypercube.
    /// Panics if the MLEs for the two matrices provided by the circuit map are of the wrong size.
    fn compute_data_outputs(
        &self,
        mle_outputs_necessary: &HashSet<&MleDescription<E>>,
        circuit_map: &mut CircuitEvalMap<E>,
    ) {
        assert_eq!(mle_outputs_necessary.len(), 1);
        let mle_output_necessary = mle_outputs_necessary.iter().next().unwrap();

        let matrix_a_data = circuit_map
            .get_data_from_circuit_mle(&self.matrix_a.mle)
            .unwrap();
        assert_eq!(
            matrix_a_data.num_vars(),
            self.matrix_a.rows_num_vars + self.matrix_a.cols_num_vars
        );

        let matrix_b_data = circuit_map
            .get_data_from_circuit_mle(&self.matrix_b.mle)
            .unwrap();
        assert_eq!(
            matrix_b_data.num_vars(),
            self.matrix_b.rows_num_vars + self.matrix_b.cols_num_vars
        );

        let product = product_two_matrices_from_flattened_vectors(
            &matrix_a_data.to_vec(),
            &matrix_b_data.to_vec(),
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
    }

    fn convert_into_verifier_layer(
        &self,
        sumcheck_bindings: &[E],
        claim_points: &[&[E]],
        transcript_reader: &mut impl VerifierTranscript<E::BaseField>,
    ) -> Result<Self::VerifierLayer> {
        // For matmult, we only use interpolative claim aggregation.
        assert_eq!(claim_points.len(), 1);
        let claim_point = claim_points[0];

        // Split the claim into the claims made on matrix A rows and matrix B cols.
        let mut claim_a = claim_point.to_vec();
        let claim_b = claim_a.split_off(self.matrix_a.rows_num_vars);

        // Construct the full claim made on A using the claim made on the layer and the sumcheck bindings.
        let full_claim_chals_a = claim_a
            .into_iter()
            .chain(sumcheck_bindings.to_vec())
            .collect_vec();

        // Construct the full claim made on B using the claim made on the layer and the sumcheck bindings.
        let full_claim_chals_b = sumcheck_bindings
            .iter()
            .copied()
            .chain(claim_b)
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

    /// Return the [PostSumcheckLayer], given challenges that fully bind the expression.
    fn get_post_sumcheck_layer(
        &self,
        round_challenges: &[E],
        claim_challenges: &[&[E]],
        _random_coefficients: &[E],
    ) -> PostSumcheckLayer<E, Option<E>> {
        // We are always using interpolative claim aggregation for MatMult layers.
        assert_eq!(claim_challenges.len(), 1);
        let claim_challenge = claim_challenges[0];
        let mut pre_bound_matrix_a_mle = self.matrix_a.mle.clone();
        let claim_chals_matrix_a = claim_challenge[..self.matrix_a.rows_num_vars].to_vec();
        let mut indexed_index_counter = 0;
        let mut bound_index_counter = 0;

        // We need to make sure the MLE indices of the post-sumcheck layer
        // match the MLE indices in proving, since it is pre-processed
        // when we start proving.
        // I.e, we keep the first variables representing the columns of matrix
        // A as Indexed for sumcheck, and keep the rest as bound to their
        // respective claim point in pre-processing.
        let matrix_a_new_indices = self
            .matrix_a
            .mle
            .mle_indices()
            .iter()
            .map(|mle_idx| match mle_idx {
                &MleIndex::Indexed(_) => {
                    if bound_index_counter < self.matrix_a.rows_num_vars {
                        let ret = MleIndex::Bound(
                            claim_chals_matrix_a[bound_index_counter],
                            bound_index_counter,
                        );
                        bound_index_counter += 1;
                        ret
                    } else {
                        let ret = MleIndex::Indexed(indexed_index_counter);
                        indexed_index_counter += 1;
                        ret
                    }
                }
                MleIndex::Fixed(_) => mle_idx.clone(),
                MleIndex::Free => panic!("should not have any free indices"),
                MleIndex::Bound(_, _) => panic!("should not have any bound indices"),
            })
            .collect_vec();
        pre_bound_matrix_a_mle.set_mle_indices(matrix_a_new_indices);

        // We keep the last variables representing the rows of matrix B
        // as Indexed for sumcheck, and keep the rest as bound to their
        // respective claim point in pre-processing.
        let mut pre_bound_matrix_b_mle = self.matrix_b.mle.clone();
        let claim_chals_matrix_b = claim_challenge[self.matrix_a.rows_num_vars..].to_vec();
        let mut bound_index_counter = 0;
        let mut indexed_index_counter = 0;
        let matrix_b_new_indices = self
            .matrix_b
            .mle
            .mle_indices()
            .iter()
            .map(|mle_idx| match mle_idx {
                &MleIndex::Indexed(_) => {
                    if indexed_index_counter < self.matrix_b.rows_num_vars {
                        let ret = MleIndex::Indexed(indexed_index_counter);
                        indexed_index_counter += 1;
                        ret
                    } else {
                        let ret = MleIndex::Bound(
                            claim_chals_matrix_b[bound_index_counter],
                            bound_index_counter,
                        );
                        bound_index_counter += 1;
                        ret
                    }
                }
                MleIndex::Fixed(_) => mle_idx.clone(),
                MleIndex::Free => panic!("should not have any free indices"),
                MleIndex::Bound(_, _) => panic!("should not have any bound indices"),
            })
            .collect_vec();
        pre_bound_matrix_b_mle.set_mle_indices(matrix_b_new_indices);
        let mles = vec![pre_bound_matrix_a_mle, pre_bound_matrix_b_mle];

        PostSumcheckLayer(vec![Product::<E, Option<E>>::new(
            &mles,
            E::ONE,
            round_challenges,
        )])
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn get_circuit_mles(&self) -> Vec<&MleDescription<E>> {
        vec![&self.matrix_a.mle, &self.matrix_b.mle]
    }

    fn convert_into_prover_layer<'a>(&self, circuit_map: &CircuitEvalMap<E>) -> LayerEnum<E> {
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
#[serde(bound = "E: ExtensionField")]
pub struct VerifierMatMultLayer<E: ExtensionField> {
    /// The layer id associated with this gate layer.
    layer_id: LayerId,

    /// The LHS Matrix to be multiplied.
    matrix_a: VerifierMatrix<E>,

    /// The RHS Matrix to be multiplied.
    matrix_b: VerifierMatrix<E>,
}

impl<E: ExtensionField> VerifierLayer<E> for VerifierMatMultLayer<E> {
    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_claims(&self) -> Result<Vec<Claim<E>>> {
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

                let claim: Claim<E> = Claim::new(
                    matrix_fixed_indices,
                    matrix_claimed_val,
                    self.layer_id,
                    matrix.mle.layer_id(),
                );
                claim
            })
            .collect_vec();

        Ok(claims)
    }
}

impl<E: ExtensionField> VerifierMatMultLayer<E> {
    fn evaluate(&self) -> E {
        self.matrix_a.mle.value() * self.matrix_b.mle.value()
    }
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

    use crate::layer::matmult::product_two_matrices_from_flattened_vectors;

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

        let res_product =
            product_two_matrices_from_flattened_vectors(&mle_vec_a, &mle_vec_b, 4, 2, 2, 2);

        let exp_product = vec![
            Fr::from(3 + 2 * 9),
            Fr::from(5 + 2 * 6),
            Fr::from(9 * 3 + 10 * 9),
            Fr::from(9 * 5 + 10 * 6),
            Fr::from(13 * 3 + 9),
            Fr::from(13 * 5 + 6),
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

        let res_product =
            product_two_matrices_from_flattened_vectors(&mle_vec_a, &mle_vec_b, 8, 4, 4, 2);

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

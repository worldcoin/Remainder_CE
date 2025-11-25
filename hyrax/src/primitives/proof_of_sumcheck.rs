use rand::{CryptoRng, RngCore};
use remainder::layer::product::PostSumcheckLayer;
use serde::{Deserialize, Serialize};
use shared_types::curves::PrimeOrderCurve;
use shared_types::ff_field;
use shared_types::pedersen::{CommittedScalar, CommittedVector, PedersenCommitter};
use shared_types::transcript::ec_transcript::ECTranscriptTrait;
use std::ops::Neg;

use crate::gkr::layer::{evaluate_committed_psl, evaluate_committed_scalar};

use super::proof_of_dot_prod::ProofOfDotProduct;

#[cfg(test)]
/// Tests for proof of sumcheck.
mod tests;

/// See Figure 1 of the Hyrax paper.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "C: PrimeOrderCurve")]
pub struct ProofOfSumcheck<C: PrimeOrderCurve> {
    /// the commitment to the purported sum
    pub sum: C,
    /// the alpha_j's, where alpha_j = Com(padded_message_j; r_alpha_j)
    pub messages: Vec<C>,
    /// the proof of dot product establishing the proof of sumcheck by squashing all the messages
    pub podp: ProofOfDotProduct<C>,
}

impl<C: PrimeOrderCurve> ProofOfSumcheck<C> {
    #[allow(clippy::too_many_arguments)]
    /// As a part of weak FS, assumes that the commitments to the products as
    /// well as commitments to prover sumcheck messages have already been added
    /// to the transcript.
    pub fn prove(
        sum: &CommittedScalar<C>,
        // the alpha_j's, where alpha_j = Com(padded_message_j; r_alpha_j)
        messages: &[CommittedVector<C>],
        // the degree of all sumcheck messages (assumed uniform)
        degree: usize,
        // The fully bound MLEs as a committed PostSumcheckLayer
        post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, CommittedScalar<C>>,
        // the verifier bindings
        bindings: &[C::Scalar],
        committer: &PedersenCommitter<C>,
        mut rng: &mut (impl CryptoRng + RngCore),
        // the transcript AFTER having sampled the verifier bindings
        transcript: &mut impl ECTranscriptTrait<C>,
    ) -> Self {
        // the number of sumcheck rounds
        let n = messages.len();

        debug_assert_eq!(n, bindings.len());

        // sample the RLC coeffs for the matrix rows, rho_1, .., rho_{n+1}, from the transcript
        let rhos = transcript.get_scalar_field_challenges(
            "Proof of sumcheck RLC coefficients for batching rows",
            n + 1,
        );
        assert_eq!(rhos.len(), n + 1);

        // sample the RLC coeffs for the alpha_j's, gamma_1, .., gamma_n, from the transcript
        let gammas = transcript.get_scalar_field_challenges(
            "Proof of sumcheck RLC coefficients for batching columns",
            n,
        );
        assert_eq!(gammas.len(), n);

        // compute alpha, the commitment to the gamma combo of the alphas (private vector in PoDP)
        let len_vector_commits = if !messages.is_empty() {
            messages[0].value.len()
        } else {
            0
        };
        let zero = CommittedVector::<C>::zero(len_vector_commits);
        let alpha: CommittedVector<C> = gammas
            .iter()
            .zip(messages.iter())
            .fold(zero, |acc, (gamma, message)| acc + message.clone() * *gamma);

        // calculate the vector j_star (the public vector in the PoDP)
        let j_star: Vec<C::Scalar> = Self::calculate_j_star(bindings, &rhos, &gammas, degree);

        // need a CommittedScalar for the result of the dot product
        let oracle_eval = evaluate_committed_scalar(post_sumcheck_layer);
        let sum_commitment = sum.commitment;
        let dot_product: CommittedScalar<C> =
            oracle_eval * rhos[rhos.len() - 1].neg() + sum.clone() * rhos[0];

        let podp = ProofOfDotProduct::<C>::prove(
            &alpha,
            &dot_product,
            &j_star,
            committer,
            &mut rng,
            transcript,
        );

        Self {
            sum: sum_commitment,
            messages: messages.iter().map(|m| m.commitment).collect(),
            podp,
        }
    }

    /// As a part of weak FS, assumes that the commitments to the products as
    /// well as prover sumcheck message commitments have already been added to
    /// the transcript.
    pub fn verify(
        &self,
        // The expected sum
        sum: &C,
        // the degree of all sumcheck messages (assumed uniform)
        degree: usize,
        // The fully bound MLEs as a PostSumcheckLayer
        post_sumcheck_layer: &PostSumcheckLayer<C::Scalar, C>,
        // the verifier bindings
        bindings: &[C::Scalar],
        committer: &PedersenCommitter<C>,
        // the transcript AFTER having sampled the verifier bindings
        transcript: &mut impl ECTranscriptTrait<C>,
    ) {
        // Verify that the purported sum is correct
        assert_eq!(self.sum, *sum);

        // the number of sumcheck rounds
        let n = self.messages.len();

        // sample the RLC coeffs for the matrix rows, rho_1, .., rho_{n+1}, from the transcript
        let rhos = transcript.get_scalar_field_challenges(
            "Proof of sumcheck RLC coefficients for batching rows",
            n + 1,
        );
        assert_eq!(rhos.len(), n + 1);

        // sample the RLC coeffs for the alpha_j's, gamma_1, .., gamma_n, from the transcript
        let gammas = transcript.get_scalar_field_challenges(
            "Proof of sumcheck RLC coefficients for batching columns",
            n,
        );
        debug_assert_eq!(gammas.len(), n);

        // compute alpha, the commitment to the gamma combo of the alphas (private vector in PoDP)
        let alpha = gammas
            .iter()
            .zip(self.messages.iter())
            .fold(C::zero(), |acc, (gamma, message)| acc + *message * *gamma);

        // calculate the vector j_star (the public vector in the PoDP)
        let j_star: Vec<C::Scalar> = Self::calculate_j_star(bindings, &rhos, &gammas, degree);

        // calculate a commitment to the expected dot product
        let oracle_eval = evaluate_committed_psl(post_sumcheck_layer);
        let dot_product: C = self.sum * rhos[0] + oracle_eval * rhos[rhos.len() - 1].neg();

        self.podp
            .verify(&alpha, &dot_product, &j_star, committer, transcript);
    }

    /// Calculate the vector j_star, which is the public vector in the proof of dot product, formed
    /// as indicated from the matrix encoding the verifier bindings: each row is multiplied by a
    /// row, and each group of columns is divided by a gamma.
    /// Returned vector has length (degree + 1) * number of rounds (number of rounds =
    /// bindings.len()).
    /// E.g. for a degree 2 sumcheck, with four rounds and verifier bindings r0, .. r3, the vector
    /// would be the sum down columns of:
    /// ```text
    ///             / gammas[0]     / gammas[1]     / gammas[2]      / gammas[3]
    /// rhos[0] * (2    1    1      0   0     0     0   0     0      0    0      0)
    /// rhos[1] * (-1   -r0  -r0^2  2   1     1     0   0     0      0    0      0)
    /// rhos[2] * (0    0    0      -1  -r1  -r1^2  2   1     1      0    0      0)
    /// rhos[3] * (0    0    0      0   0      0    -1  -r2  -r2^2   2    1      1)
    /// rhos[4] * (0    0    0      0   0      0    0   0     0      -1  -r3   -r3^2)
    /// ```
    pub fn calculate_j_star(
        // the verifier bindings during sumcheck
        bindings: &[C::Scalar],
        // the coefficients of the random linear combination of the rows of the matrix
        rhos: &[C::Scalar],
        // the coefficients of the random linear combination of the padded sumcheck messages
        gammas: &[C::Scalar],
        // the degree of the sumcheck messages
        degree: usize,
    ) -> Vec<C::Scalar> {
        let n_rounds = bindings.len();
        assert_eq!(rhos.len(), n_rounds + 1);
        assert_eq!(gammas.len(), n_rounds);
        let mut j_star = vec![C::Scalar::ZERO; (degree + 1) * n_rounds];
        for round in 0..n_rounds {
            let gamma_inv = gammas[round].invert().unwrap();
            let mut power: C::Scalar = C::Scalar::ONE;
            for d in 0..degree + 1 {
                let i = round * (degree + 1) + d;
                let coeff = if d == 0 {
                    C::Scalar::from(2)
                } else {
                    C::Scalar::ONE
                };
                j_star[i] = gamma_inv * (rhos[round] * coeff - rhos[round + 1] * power);
                power *= bindings[round];
            }
        }
        j_star
    }
}

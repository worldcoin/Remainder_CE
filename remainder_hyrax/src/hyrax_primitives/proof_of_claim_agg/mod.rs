use crate::hyrax_gkr::hyrax_layer::HyraxClaim;
use crate::pedersen::{CommittedScalar, PedersenCommitter};
use rand::Rng;
use remainder::layer::LayerId;
use remainder_shared_types::curves::PrimeOrderCurve;
use remainder_shared_types::ff_field;
use remainder_shared_types::transcript::ec_transcript::{ECProverTranscript, ECVerifierTranscript};
use remainder_shared_types::Field;

use super::proof_of_equality::ProofOfEquality;
use super::proof_of_opening::ProofOfOpening;

pub mod tests;

pub struct ProofOfClaimAggregation<C: PrimeOrderCurve> {
    // the commitments to the coefficients of the interpolating polynomial
    pub interpolant_coeffs: Vec<C>,
    // proofs of opening for the hs
    pub proofs_of_opening: Vec<ProofOfOpening<C>>,
    // the proofs of equality between the evaluations and the evaluations of the polynomial (in that order)
    pub proofs_of_equality: Vec<ProofOfEquality<C>>,
}

impl<C: PrimeOrderCurve> ProofOfClaimAggregation<C> {
    /// Returns an instance of ProofOfClaimAggregation, the evaluation point for the aggregate
    /// claim, and the commitment to its evaluation.
    pub fn prove(
        // the claims to be aggregated (must all be to the same layer)
        claims: &[HyraxClaim<C::Scalar, CommittedScalar<C>>],
        // the coefficients of the interpolating polynomial V_i(l(x))
        interpolant_coeffs: &[C::Scalar],
        committer: &PedersenCommitter<C>,
        mut rng: &mut impl Rng,
        transcript: &mut impl ECProverTranscript<C>,
    ) -> (Self, HyraxClaim<C::Scalar, CommittedScalar<C>>) {
        // Check that all claims are on the same layer
        let layer_id = claims[0].to_layer_id;
        assert!(claims.iter().all(|claim| claim.to_layer_id == layer_id));

        // Sample blinding factors for committing to the coefficients of the polynomial (from the rng)
        let coeffs_blinding: Vec<C::Scalar> = (0..interpolant_coeffs.len())
            .map(|_| C::Scalar::random(&mut rng))
            .collect();

        // Compute commitments $Com(H_i) $ to the coefficients of this polynomial
        let coeffs: Vec<CommittedScalar<C>> = interpolant_coeffs
            .iter()
            .zip(coeffs_blinding.iter())
            .map(|(h, blinding)| committer.committed_scalar(h, blinding))
            .collect();

        // Add the commitments to the coefficients to the transcript
        coeffs.iter().enumerate().for_each(|(i, commit_triple)| {
            let label = format!("PROVER coeff {}", i);
            transcript.append_ec_point(Box::leak(label.into_boxed_str()), commit_triple.commitment);
        });

        // Perform a proof of opening for each commitment $Com(H_i)$
        let proofs_of_opening: Vec<ProofOfOpening<C>> = coeffs
            .iter()
            .map(|commit_triple| {
                ProofOfOpening::prove(commit_triple, committer, &mut rng, transcript)
            })
            .collect();

        // Generate a proof of equality for each claim $q^{(j)} \mapsto t_j$ being aggregated
        let proofs_of_equality: Vec<ProofOfEquality<C>> = claims
            .iter()
            .map(|claim| claim.evaluation.clone())
            .enumerate()
            .map(|(j, evaluation)| {
                // Compute $f(j)$
                let mut power = C::Scalar::ONE;
                let f_j: CommittedScalar<C> =
                    coeffs.iter().fold(CommittedScalar::zero(), |acc, coeff| {
                        let next_acc = acc + coeff.clone() * power;
                        power *= C::Scalar::from(j as u64);
                        next_acc
                    });

                // Perform a proof of equality on commitments to f(j) and the value of the jth claim
                ProofOfEquality::prove(&evaluation, &f_j, committer, &mut rng, transcript)
            })
            .collect();

        // A random evaluation point $\tau \in \mathbb{F}$ is sampled from the transcript
        let tau = transcript.get_scalar_field_challenge("PoCA tau");

        // Calculate the powers of tau
        let powers_of_tau: Vec<C::Scalar> =
            (0..coeffs.len()).fold(vec![C::Scalar::ONE], |mut acc, _| {
                acc.push(*acc.last().unwrap() * tau);
                acc
            });

        // Compute the aggregate claim
        let claim_points: Vec<_> = claims.iter().map(|claim| claim.point.clone()).collect();
        let agg_claim_point = Self::interpolate_evaluation_vector(&claim_points, tau);
        // Compute the commitment to the evaluation of the interpolating polynomial using the commitments to the coefficients.
        let agg_claim_eval = coeffs.iter().zip(powers_of_tau.iter()).fold(
            CommittedScalar::<C>::zero(),
            |acc, (coeff_commit, power_of_tau)| acc + coeff_commit.clone() * *power_of_tau,
        );
        let agg_claim = HyraxClaim {
            to_layer_id: layer_id,
            point: agg_claim_point,
            mle_enum: None,
            evaluation: agg_claim_eval,
        };

        (
            ProofOfClaimAggregation {
                interpolant_coeffs: coeffs.iter().map(|coeff| coeff.commitment).collect(),
                proofs_of_opening,
                proofs_of_equality,
            },
            agg_claim,
        )
    }

    /// Returns the evaluation point of the aggregate claim and the commitment to its evaluation.
    pub fn verify(
        &self,
        // the claims: each being a point (in the clear) and the commitment to the evaluation
        claims: &[HyraxClaim<C::Scalar, C>],
        committer: &PedersenCommitter<C>,
        transcript: &mut impl ECVerifierTranscript<C>,
    ) -> HyraxClaim<C::Scalar, C> {
        // Check the there are the correct number of coefficient commitments (i.e. check the degree
        // of the interpolating polynomial)
        let n_claims = claims.len();
        let n_bits = claims[0].point.len();
        let expected_num_coeffs = (n_claims - 1) * n_bits + 1;
        let layer_id: LayerId = claims[0].to_layer_id;
        assert_eq!(expected_num_coeffs, self.interpolant_coeffs.len());

        // Check that all claims are on the same layer
        assert!(claims.iter().all(|claim| claim.to_layer_id == layer_id));

        // Add the commitments to the coefficients to the transcript
        self.interpolant_coeffs
            .iter()
            .enumerate()
            .for_each(|(i, commit)| {
                let label = format!("coeff {}", i);

                let transcript_commit = transcript
                    .consume_ec_point(Box::leak(label.into_boxed_str()))
                    .unwrap();
                assert_eq!(*commit, transcript_commit);
            });

        // Verify the proofs of opening for coefficients of the interpolating polynomial
        // TODO(vishady) riad audit comments: make sure to replace the asserts with errors
        assert_eq!(self.proofs_of_opening.len(), self.interpolant_coeffs.len());
        self.proofs_of_opening
            .iter()
            .zip(self.interpolant_coeffs.iter())
            .for_each(|(proof, commit_h)| {
                proof.verify(*commit_h, committer, transcript);
            });

        // Verify the proofs of equality f(j) = jth mle evaluation
        assert_eq!(self.proofs_of_equality.len(), claims.len());
        claims
            .iter()
            .map(|claim| claim.evaluation)
            .zip(self.proofs_of_equality.iter())
            .enumerate()
            .for_each(|(j, (mle_eval, proof))| {
                // calculate the commitment to f(j) using the commitments to the coefficients
                let mut power = C::Scalar::ONE;
                let commit_f_j = self
                    .interpolant_coeffs
                    .iter()
                    .fold(C::zero(), |acc, commit_h| {
                        let next_acc = acc + *commit_h * power;
                        power *= C::Scalar::from(j as u64);
                        next_acc
                    });
                proof.verify(mle_eval, commit_f_j, committer, transcript);
            });

        // A random evaluation point $\tau \in \mathbb{F}$ is sampled from the transcript
        let tau = transcript.get_scalar_field_challenge("PoCA tau").unwrap();

        // Calculate the powers of tau
        let powers_of_tau: Vec<C::Scalar> =
            (0..self.interpolant_coeffs.len()).fold(vec![C::Scalar::ONE], |mut acc, _| {
                acc.push(*acc.last().unwrap() * tau);
                acc
            });

        let claim_points: Vec<_> = claims.iter().map(|claim| claim.point.clone()).collect();
        // The evaluation point of the aggregate claim.
        let agg_claim_point = Self::interpolate_evaluation_vector(&claim_points, tau);

        // Compute the commitment $a$ to the evaluation of the interpolating polynomial using the
        // commitments to the coefficients.
        let agg_claim_eval = self
            .interpolant_coeffs
            .iter()
            .zip(powers_of_tau.iter())
            .fold(C::zero(), |acc, (commit_h, power_of_tau)| {
                acc + *commit_h * *power_of_tau
            });

        HyraxClaim {
            to_layer_id: layer_id,
            point: agg_claim_point,
            mle_enum: None,
            evaluation: agg_claim_eval,
        }
    }

    /// Return the evaluation point of the aggregate claim, where `tau` is the point on the
    /// polynomial function interpolating between them.
    fn interpolate_evaluation_vector(
        claim_points: &[Vec<C::Scalar>],
        tau: C::Scalar,
    ) -> Vec<C::Scalar> {
        let weights = barycentric_weights(tau, claim_points.len());
        let mut claim_point = vec![C::Scalar::ZERO; claim_points[0].len()];
        for i in 0..claim_points[0].len() {
            for j in 0..claim_points.len() {
                claim_point[i] += claim_points[j][i] * weights[j];
            }
        }
        claim_point
    }
}

// TODO we stole the content of this function from `evaluate_at_point` in `remainder_prover`
// and place it here to avoid a circular dependency.  Note that when integrating with production
// remainder, it would make sense to remove `evaluate_at_point` from `remainder_prover` and call
// this function instead, in order to avoid repeatedly recomputing the barycentric weights.

/// Return weights that can be used to interpolate a polynomial at a given point.
/// Specifically, if f(0), .. , f(n_evals) are the evaluations of a polynomial f,
/// and `weights` is the return value of this function, then the evaluation of f at `point` is
/// `f(point) = sum(f(i) * weights[i])`.
pub fn barycentric_weights<F: Field>(point: F, n_evals: usize) -> Vec<F> {
    (0..n_evals)
        .map(
            // Create an iterator of everything except current value
            |x| {
                (0..x)
                    .chain(x + 1..n_evals)
                    .map(|x| F::from(x as u64))
                    .fold(
                        // Compute vector of (numerator, denominator)
                        (F::ONE, F::ONE),
                        |(num, denom), val| {
                            (num * (point - val), denom * (F::from(x as u64) - val))
                        },
                    )
            },
        )
        .map(|(num, denom)| num * denom.invert().unwrap())
        .collect()
}

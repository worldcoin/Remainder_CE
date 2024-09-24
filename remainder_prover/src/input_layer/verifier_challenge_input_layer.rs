//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::VerifierTranscript,
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::Claim,
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use crate::mle::Mle;
use thiserror::Error;

#[derive(Error, Clone, Debug)]
/// The errors which can be encountered when constructing an input layer.
pub enum VerifierChallengeError {
    /// This is when the random input layer evaluated at a random point does not
    /// equal the claimed value.
    #[error("The evaluation point of the claim is too short")]
    InsufficientBinding,
    #[error("Evaluation of MLE does not match the claimed value")]
    EvaluationMismatch,
}

/// Represents a verifier challenge, where we generate random constants in the
/// form of coefficients of an MLE that can be used e.g. for packing constants, or in logup, or
/// permutation checks and so on.
#[derive(Debug, Clone)]
pub struct VerifierChallenge<F: Field> {
    /// The data.
    mle: MultilinearExtension<F>,
    /// The layer ID.
    pub(crate) layer_id: LayerId,
}

/// Verifier's description of a [VerifierChallenge].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: Field")]
pub struct CircuitVerifierChallenge<F: Field> {
    /// The layer ID.
    layer_id: LayerId,

    /// The number of variables needed to index the data of this verifier challenge.
    pub num_bits: usize,

    _marker: PhantomData<F>,
}

impl<F: Field> CircuitVerifierChallenge<F> {
    /// Constructor for the [CircuitVerifierChallenge] using the
    /// number of bits that are in the MLE of the layer.
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> VerifierChallenge<F> {
    /// Constructor for the [VerifierChallenge] using the layer_id
    /// and the MLE that is stored in this input layer.
    pub fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }

    /// Return the MLE stored in self as a DenseMle with the correct layer ID.
    pub fn get_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }

    /// Return the layer id.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Evaluate the MLE at the claim point.
    /// Panics if the claim point is not the correct length.
    pub fn evaluate(
        &self,
        point: &[F]
    ) -> Result<F, VerifierChallengeError> {
        let mle_evals = self.mle.get_evals_vector().clone();
        let mut mle_ref = DenseMle::<F>::new_from_raw(mle_evals, self.layer_id);
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_iterated_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in point.iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            eval.ok_or(VerifierChallengeError::InsufficientBinding)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };
        Ok(eval.get_result())
    }

    // FIXME(Ben) - rewrite this to use evaluate
    /// On a copy of the underlying data, fix variables for each of the coordinates of the the point
    /// in `claim`, and check whether the single element left in the bookkeeping table is equal to
    /// the claimed value in `claim`.
    pub fn verify(
        &self,
        claim: &Claim<F>,
    ) -> Result<(), VerifierChallengeError> {
        let mle_evals = self.mle.get_evals_vector().clone();
        let mut mle_ref = DenseMle::<F>::new_from_raw(mle_evals, self.layer_id);
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_iterated_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            eval.ok_or(VerifierChallengeError::InsufficientBinding)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };

        // This would be an internal error and should never happen.
        assert_eq!(eval.get_point(), claim.get_point());

        // Check if the evaluation of the MLE matches the claimed value.
        if eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            Err(VerifierChallengeError::EvaluationMismatch)
        }
    }

}

impl<F: Field> CircuitVerifierChallenge<F> {
    /// Return the layer id.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Read the data from the transcript (in the interactive setting, this is just asking the
    /// verifier for the appropriate number of challenges).
    pub fn get_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> VerifierChallenge<F> {
        let num_evals = 1 << self.num_bits;
        let values = transcript_reader.get_challenges("Verifier challenges", num_evals).unwrap();
        VerifierChallenge::new(
            MultilinearExtension::new(values),
            self.layer_id,
        )
    }

    // FIXME come back to this - might want to combine with get_from_transcript
    pub fn instantiate(&self, values: Vec<F>) -> VerifierChallenge<F> {
        VerifierChallenge::new(
            MultilinearExtension::new(values),
            self.layer_id,
        )
    }

    // FIXME come back to this - might want to combine with get_from_transcript
    fn convert_into_prover_version(
        &self,
        combined_mle: MultilinearExtension<F>,
    ) -> VerifierChallenge<F> {
        VerifierChallenge::new(combined_mle, self.layer_id)
    }
}

#[cfg(test)]
mod tests {
    use remainder_shared_types::ff_field;
    use remainder_shared_types::{
        transcript::{test_transcript::TestSponge, TranscriptReader, TranscriptWriter, ProverTranscript},
        Fr,
    };

    use super::*;

    #[test]
    fn test_circuit_verifier_challenge() {
        // Setup phase.
        let layer_id = LayerId::Input(0);

        // MLE on 2 variables.
        let num_vars = 2;
        let num_evals = 1 << num_vars;

        // Transcript writer with test sponge that always returns `1`.
        let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
            TranscriptWriter::new("Test Transcript Writer");

        let claim_point = vec![Fr::ONE, Fr::ZERO];
        let claim_result = Fr::from(1);
        let claim: Claim<Fr> = Claim::new(claim_point, claim_result);

        let mle_vec = transcript_writer.get_challenges("random challenges for FS", num_evals);
        let mle = MultilinearExtension::new(mle_vec);

        let verifier_challenge_description =
            CircuitVerifierChallenge::<Fr>::new(layer_id, mle.num_vars());
        // Nothing really to test for VerifierChallenge
        let _verifier_challenge = VerifierChallenge::new(mle, layer_id);

        // Verifier phase.
        // 1. Retrieve proof/transcript.
        let transcript = transcript_writer.get_transcript();
        let mut transcript_reader: TranscriptReader<Fr, TestSponge<Fr>> =
            TranscriptReader::new(transcript);

        // 2. Get commitment from transcript.
        let verifier_challenge = verifier_challenge_description.get_from_transcript(&mut transcript_reader);

        // 3. ... [skip] verify other layers.

        // 4. Verify this layer's commitment.
        verifier_challenge
            .verify(&claim)
            .unwrap();
    }
}

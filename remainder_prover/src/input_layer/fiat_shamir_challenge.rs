//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

use crate::{
    claims::RawClaim,
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use crate::mle::Mle;
use thiserror::Error;

use anyhow::{Context, Ok, Result};

#[derive(Error, Clone, Debug)]
/// The errors which can be encountered when constructing an input layer.
pub enum FiatShamirChallengeError {
    /// The point of the claim is too short.
    #[error("Claim binds {0} variables, but MLE has only {1} variables")]
    NumVarsMismatch(usize, usize),
    /// The [FiatShamirChallenge] evaluation does not equal the claimed value.
    #[error("Evaluation of MLE does not match the claimed value")]
    EvaluationMismatch,
}

/// Represents a verifier challenge, where we generate random constants in the
/// form of coefficients of an MLE that can be used e.g. for packing constants, or in logup, or
/// permutation checks and so on.
#[derive(Debug, Clone)]
pub struct FiatShamirChallenge<F: Field> {
    /// The data.
    pub mle: MultilinearExtension<F>,
    /// The layer ID.
    pub(crate) layer_id: LayerId,
}

/// Verifier's description of a [FiatShamirChallenge].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Hash)]
#[serde(bound = "F: Field")]
pub struct FiatShamirChallengeDescription<F: Field> {
    /// The layer ID.
    layer_id: LayerId,

    /// The number of variables needed to index the data of this verifier challenge.
    pub num_bits: usize,

    _marker: PhantomData<F>,
}

impl<F: Field> FiatShamirChallengeDescription<F> {
    /// Constructor for the [CircuitFiatShamirChallenge] using the
    /// number of bits that are in the MLE of the layer.
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> FiatShamirChallenge<F> {
    /// Create a new [FiatShamirChallenge] from the given MLE allocating the next available FS layer
    /// ID.
    pub fn new(mle: MultilinearExtension<F>) -> Self {
        let layer_id = LayerId::next_fiat_shamir_challenge_layer_id();
        Self { mle, layer_id }
    }

    /// Return the MLE stored in self as a DenseMle with the correct layer ID.
    pub fn get_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.to_vec(), self.layer_id)
    }

    /// Return the layer id.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// On a copy of the underlying data, fix variables for each of the coordinates of the the point
    /// in `claim`, and check whether the single element left in the bookkeeping table is equal to
    /// the claimed value in `claim`.
    pub fn verify(&self, claim: &RawClaim<F>) -> Result<()> {
        let mle_evals = self.mle.to_vec();
        let mut mle = DenseMle::<F>::new_from_raw(mle_evals, self.layer_id);
        mle.index_mle_indices(0);

        if mle.num_free_vars() != claim.get_point().len() {
            return Err(FiatShamirChallengeError::NumVarsMismatch(
                claim.get_point().len(),
                mle.num_free_vars(),
            ))
            .with_context(|| "");
        }

        let eval = if mle.num_free_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle.fix_variable(curr_bit, chal);
            }
            eval.unwrap()
        } else {
            RawClaim::new(vec![], mle.value())
        };

        // This is an internal error as it should never happen.
        assert_eq!(eval.get_point(), claim.get_point());

        // Check if the evaluation of the MLE matches the claimed value.
        if eval.get_eval() == claim.get_eval() {
            Ok(())
        } else {
            Err(FiatShamirChallengeError::EvaluationMismatch).with_context(|| "")
        }
    }
}

impl<F: Field> FiatShamirChallengeDescription<F> {
    /// Return the layer id.
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    /// Create a [FiatShamirChallenge] from this [CircuitFiatShamirChallenge] and the given values.
    /// Panics if the length of `values` is not equal to the number of evaluations in the MLE.
    pub fn instantiate(&self, values: Vec<F>) -> FiatShamirChallenge<F> {
        assert_eq!(values.len(), 1 << self.num_bits);
        FiatShamirChallenge {
            mle: MultilinearExtension::new(values),
            layer_id: self.layer_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use remainder_shared_types::ff_field;
    use remainder_shared_types::{
        transcript::{
            test_transcript::TestSponge, ProverTranscript, TranscriptReader, TranscriptWriter,
            VerifierTranscript,
        },
        Fr,
    };

    #[test]
    fn test_circuit_fiat_shamir_challenge() {
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
        let claim: RawClaim<Fr> = RawClaim::new(claim_point, claim_result);

        let mle_vec = transcript_writer.get_challenges("random challenges for FS", num_evals);
        let mle = MultilinearExtension::new(mle_vec);

        let fs_desc = FiatShamirChallengeDescription::<Fr>::new(layer_id, mle.num_vars());
        // Nothing really to test for FiatShamirChallenge
        let _fiat_shamir_challenge = FiatShamirChallenge::new(mle);

        // Verifier phase.
        // 1. Retrieve proof/transcript.
        let transcript = transcript_writer.get_transcript();
        let mut transcript_reader: TranscriptReader<Fr, TestSponge<Fr>> =
            TranscriptReader::new(transcript);

        // 2. Get commitment from transcript.
        let values = transcript_reader
            .get_challenges("Verifier challenges", 1 << fs_desc.num_bits)
            .unwrap();
        let fiat_shamir_challenge = fs_desc.instantiate(values);

        // 3. ... [skip] verify other layers.

        // 4. Verify this layer's commitment.
        fiat_shamir_challenge.verify(&claim).unwrap();
    }
}

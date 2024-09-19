//! A part of the input layer that is random and secured through F-S

use std::marker::PhantomData;

use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::{wlx_eval::YieldWLXEvals, Claim},
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension, mle_enum::MleEnum},
};

use super::{
    enum_input_layer::InputLayerEnum, get_wlx_evaluations_helper, CircuitInputLayer,
    CommitmentEnum, InputLayer, InputLayerError,
};
use crate::mle::Mle;

/// Represents a random input layer, where we generate random constants in the
/// form of coefficients of an MLE that we can use for packing constants.

#[derive(Debug, Clone)]
pub struct VerifierChallengeInputLayer<F: Field> {
    mle: MultilinearExtension<F>,
    pub(crate) layer_id: LayerId,
}

/// Verifier's description of a [VerifierChallengeInputLayer].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: Field")]
pub struct CircuitVerifierChallengeInputLayer<F: Field> {
    /// The ID of this Random Input Layer.
    layer_id: LayerId,

    /// The number of variables this Random Input Layer is on.
    pub num_bits: usize,

    _marker: PhantomData<F>,
}

impl<F: Field> CircuitVerifierChallengeInputLayer<F> {
    /// Constructor for the [CircuitVerifierChallengeInputLayer] using the
    /// number of bits that are in the MLE of the layer.
    pub fn new(layer_id: LayerId, num_bits: usize) -> Self {
        Self {
            layer_id,
            num_bits,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> InputLayer<F> for VerifierChallengeInputLayer<F> {
    type ProverCommitment = Vec<F>;
    type VerifierCommitment = Vec<F>;

    fn commit(&mut self) -> Result<Self::VerifierCommitment, super::InputLayerError> {
        // We do not need to commit to the randomness, so we simply send it in
        // the clear.
        Ok(self.mle.get_evals_vector().clone())
    }

    fn append_commitment_to_transcript(
        commitment: &Self::VerifierCommitment,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        transcript_writer.append_elements("Random Layer Evaluations", commitment);
    }

    fn open(
        &self,
        _transcript: &mut impl ProverTranscript<F>,
        _claim: Claim<F>,
    ) -> Result<(), super::InputLayerError> {
        // We do not have an opening proof because we did not commit to
        // anything. The MLE exists in the clear.
        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }
}

impl<F: Field> CircuitInputLayer<F> for CircuitVerifierChallengeInputLayer<F> {
    type Commitment = Vec<F>;

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError> {
        let num_evals = 1 << self.num_bits;
        Ok(transcript_reader.get_challenges("Random Layer Evaluations", num_evals)?)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: Claim<F>,
        _transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError> {
        // In order to verify, simply fix variable on each of the variables for
        // the point in `claim`. Check whether the single element left in the
        // bookkeeping table is equal to the claimed value in `claim`.
        let mle_evals = commitment.clone();
        let mut mle_ref = DenseMle::<F>::new_from_raw(mle_evals, self.layer_id);
        mle_ref.index_mle_indices(0);

        let eval = if mle_ref.num_iterated_vars() != 0 {
            let mut eval = None;
            for (curr_bit, &chal) in claim.get_point().iter().enumerate() {
                eval = mle_ref.fix_variable(curr_bit, chal);
            }
            eval.ok_or(InputLayerError::RandomInputVerificationFailed)?
        } else {
            Claim::new(vec![], mle_ref.current_mle[0])
        };

        if eval.get_point() == claim.get_point() && eval.get_result() == claim.get_result() {
            Ok(())
        } else {
            Err(InputLayerError::RandomInputVerificationFailed)
        }
    }

    fn into_prover_input_layer(
        &self,
        combined_mle: MultilinearExtension<F>,
        precommit: &Option<CommitmentEnum<F>>,
    ) -> InputLayerEnum<F> {
        assert!(
            precommit.is_none(),
            "Verifier challenge input layer does not support precommit!"
        );
        let verifier_challenge_input_layer =
            VerifierChallengeInputLayer::new(combined_mle, self.layer_id);
        verifier_challenge_input_layer.into()
    }
}

impl<F: Field> VerifierChallengeInputLayer<F> {
    /// Constructor for the [VerifierChallengeInputLayer] using the layer_id
    /// and the MLE that is stored in this input layer.
    pub fn new(mle: MultilinearExtension<F>, layer_id: LayerId) -> Self {
        Self { mle, layer_id }
    }

    /// Return the MLE stored in self as a DenseMle with the correct layer ID.
    pub fn get_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }
}

impl<F: Field> YieldWLXEvals<F> for VerifierChallengeInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        get_wlx_evaluations_helper(
            self.mle.clone(),
            claim_vecs,
            claimed_vals,
            claimed_mles,
            num_claims,
            num_idx,
        )
    }
}

#[cfg(test)]
mod tests {
    use remainder_shared_types::ff_field;
    use remainder_shared_types::{
        transcript::{test_transcript::TestSponge, TranscriptReader, TranscriptWriter},
        Fr,
    };

    use super::*;

    #[test]
    fn test_into_verifier_random_input_layer() {
        let layer_id = LayerId::Input(0);

        let num_vars = 2;
        let num_evals = 1 << num_vars;

        // Transcript writer with test sponge that always returns `1`.
        let mut transcript_writer: TranscriptWriter<Fr, TestSponge<Fr>> =
            TranscriptWriter::new("Test Transcript Writer");

        let mle_vec = transcript_writer.get_challenges("random challenges for FS", num_evals);
        let mle = MultilinearExtension::new(mle_vec);

        let verifier_random_input_layer =
            CircuitVerifierChallengeInputLayer::<Fr>::new(layer_id, mle.num_vars());
        let random_input_layer = VerifierChallengeInputLayer::new(mle, layer_id);

        let expected_verifier_random_input_layer =
            CircuitVerifierChallengeInputLayer::new(layer_id, num_vars);

        assert_eq!(
            verifier_random_input_layer,
            expected_verifier_random_input_layer
        );
    }

    #[test]
    fn test_random_input_layer() {
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

        let verifier_random_input_layer =
            CircuitVerifierChallengeInputLayer::<Fr>::new(layer_id, mle.num_vars());
        let mut random_input_layer = VerifierChallengeInputLayer::new(mle, layer_id);

        // Prover phase.
        // 1. Commit to the input layer.
        let commitment = random_input_layer.commit().unwrap();

        // 2. Add commitment to transcript.
        VerifierChallengeInputLayer::<Fr>::append_commitment_to_transcript(
            &commitment,
            &mut transcript_writer,
        );

        // 3. ... [skip] proving other layers ...

        // 4. Open commitment (no-op for Public Layers).
        random_input_layer
            .open(&mut transcript_writer, claim.clone())
            .unwrap();

        // Verifier phase.
        // 1. Retrieve proof/transcript.
        let transcript = transcript_writer.get_transcript();
        let mut transcript_reader: TranscriptReader<Fr, TestSponge<Fr>> =
            TranscriptReader::new(transcript);

        // 2. Get commitment from transcript.
        let commitment = verifier_random_input_layer
            .get_commitment_from_transcript(&mut transcript_reader)
            .unwrap();

        // 3. ... [skip] verify other layers.

        // 4. Verify this layer's commitment.
        verifier_random_input_layer
            .verify(&commitment, claim, &mut transcript_reader)
            .unwrap();
    }
}

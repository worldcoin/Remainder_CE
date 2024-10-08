//! An InputLayer that will have it's claim proven with a Ligero Opening Proof.

use std::marker::PhantomData;

use remainder_ligero::{
    ligero_commit::{
        remainder_ligero_commit, remainder_ligero_eval_prove, remainder_ligero_verify,
    },
    ligero_structs::LigeroAuxInfo,
    poseidon_ligero::PoseidonSpongeHasher,
    LcCommit, LcRoot,
};
use remainder_shared_types::{
    transcript::{ProverTranscript, VerifierTranscript},
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::wlx_eval::YieldWLXEvals,
    input_layer::CommitmentEnum,
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use super::{
    enum_input_layer::InputLayerEnum, get_wlx_evaluations_helper, InputLayerTrait,
    InputLayerDescriptionTrait, InputLayerError,
};

/// An input layer in which `mle` will be committed to using the Ligero polynomial
/// commitment scheme.
#[derive(Debug)]
pub struct LigeroInputLayer<F: Field> {
    /// The MLE which we wish to commit to.
    pub mle: MultilinearExtension<F>,
    /// The ID corresponding to this layer.
    pub(crate) layer_id: LayerId,

    /// The Ligero commitment to `mle`.
    comm: Option<LcCommit<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>>,

    /// The auxiliary information needed in order to perform an opening proof.
    aux: LigeroAuxInfo<F>,
}

// FIXME(Ben) not used currently
/// The Ligero evaluation proof that the prover needs to send to the verifier.
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct LigeroInputProof<F: Field> {
    /// The auxiliary information required to generate a Ligero proof.
    pub aux: LigeroAuxInfo<F>,
}

/// The Ligero commitment the prover sees, which contains more information than the verifier should see.
pub type LigeroCommitment<F> = LcCommit<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>;
/// The Ligero commitment the prover sends the verifier (adds to transcript) which is the commitment to the root.
pub type LigeroRoot<F> = LcRoot<LigeroAuxInfo<F>, F>;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(bound = "F: Field")]
/// The circuit description of a [LigeroInputLayer]. Stores the shape information of this layer.
pub struct LigeroInputLayerDescription<F: Field> {
    /// The ID of this Ligero Input Layer.
    layer_id: LayerId,

    /// The number of variables this Ligero Input Layer is on.
    num_bits: usize,

    /// The auxiliary information needed to verify the proof.
    aux: LigeroAuxInfo<F>,
}

impl<F: Field> LigeroInputLayerDescription<F> {
    /// Constructor for the [LigeroInputLayerDescription] using layer_id, num_bits
    /// which is the number of variables in the underlying MLE, and auxiliary
    /// information, which is [LigeroAuxInfo] and includes information about
    /// the encoded num rows, num cols, of the matrix of coefficients and rho_inv
    /// used for encoding.
    pub fn new(layer_id: LayerId, num_bits: usize, aux: LigeroAuxInfo<F>) -> Self {
        Self {
            layer_id,
            num_bits,
            aux,
        }
    }

    /// Return the [LigeroAuxInfo] for this layer.
    pub fn aux(&self) -> &LigeroAuxInfo<F> {
        &self.aux
    }
}

impl<F: Field> InputLayerTrait<F> for LigeroInputLayer<F> {
    type ProverCommitment = LigeroCommitment<F>;
    type VerifierCommitment = LigeroRoot<F>;

    fn commit(&mut self) -> Result<Self::VerifierCommitment, super::InputLayerError> {
        // If we've already generated a commitment (i.e. through `new_with_ligero_commitment()`),
        // there is no need to regenerate it.
        if let Some(prover_side_commit) = &self.comm {
            return Ok(prover_side_commit.get_root());
        }

        let (comm, root) = remainder_ligero_commit(self.mle.get_evals_vector(), &self.aux);

        self.comm = Some(comm);

        Ok(root)
    }

    /// Add the commitment to the prover transcript for Fiat-Shamir.
    fn append_commitment_to_transcript(
        commitment: &Self::VerifierCommitment,
        transcript_writer: &mut impl ProverTranscript<F>,
    ) {
        transcript_writer.append("Ligero Merkle Commitment", commitment.clone().into_raw());
    }

    /// "Open" the commitment, in other words, see whether the polynomial evaluated at the
    /// random point in `claim` corresopnds to the claimed value in `claim` by "opening"
    /// the commitment and this random point.
    fn open(
        &self,
        transcript_writer: &mut impl ProverTranscript<F>,
        claim: crate::claims::Claim<F>,
    ) -> Result<(), InputLayerError> {
        let comm = self
            .comm
            .clone()
            .ok_or(InputLayerError::OpeningBeforeCommitment)?;

        let _ = remainder_ligero_eval_prove(
            self.mle.get_evals_vector(),
            claim.get_point(),
            transcript_writer,
            &self.aux,
            &comm,
        );

        Ok(())
    }

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F> {
        DenseMle::new_from_raw(self.mle.get_evals_vector().clone(), self.layer_id)
    }
}

impl<F: Field> InputLayerDescriptionTrait<F> for LigeroInputLayerDescription<F> {
    type Commitment = LigeroRoot<F>;

    fn layer_id(&self) -> LayerId {
        self.layer_id
    }

    fn get_commitment_from_transcript(
        &self,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<Self::Commitment, InputLayerError> {
        let root = transcript_reader.consume_element("Ligero Merkle Commitment")?;
        Ok(Self::Commitment::new(root))
    }

    /// Verify the evaluation proof generated from the `open()` function.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        claim: crate::claims::Claim<F>,
        transcript_reader: &mut impl VerifierTranscript<F>,
    ) -> Result<(), InputLayerError> {
        // let num_coeffs = 2_usize.pow(claim.get_num_vars() as u32);
        let ligero_aux = &self.aux;
        remainder_ligero_verify::<F>(
            &commitment.root,
            ligero_aux,
            transcript_reader,
            claim.get_point(),
            claim.get_result(),
        );
        Ok(())
    }

    fn convert_into_prover_input_layer(
        &self,
        combined_mle: MultilinearExtension<F>,
        precommit: &Option<CommitmentEnum<F>>,
    ) -> InputLayerEnum<F> {
        let prover_ligero_layer = if let Some(CommitmentEnum::LigeroCommitment(ligero_commit)) =
            &precommit
        {
            LigeroInputLayer::new(
                combined_mle,
                self.layer_id,
                Some(ligero_commit.clone()),
                self.aux.rho_inv,
                (self.aux.orig_num_cols as f64) / (self.aux.num_rows as f64),
            )
        } else if precommit.is_none() {
            LigeroInputLayer::new(
                combined_mle,
                self.layer_id,
                None,
                self.aux.rho_inv,
                (self.aux.orig_num_cols as f64) / (self.aux.num_rows as f64),
            )
        } else {
            panic!("The commitment type needs to be a LigeroCommitment for a Ligero Input Layer!")
        };

        prover_ligero_layer.into()
    }
}

impl<F: Field> LigeroInputLayer<F> {
    /// Creates a new Ligero input layer depending on whether there is auxiliary information
    /// or a precommitment.
    pub fn new(
        mle: MultilinearExtension<F>,
        layer_id: LayerId,
        ligero_precomm_prover: Option<LigeroCommitment<F>>,
        rho_inv: u8,
        ratio: f64,
    ) -> Self {
        let aux = LigeroAuxInfo::new(mle.f.len().next_power_of_two(), rho_inv, ratio, None);

        Self {
            mle,
            layer_id,
            comm: ligero_precomm_prover,
            aux,
        }
    }
}

impl<F: Field> YieldWLXEvals<F> for LigeroInputLayer<F> {
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &[Vec<F>],
        claimed_vals: &[F],
        claimed_mles: Vec<DenseMle<F>>,
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
    use remainder_shared_types::{
        transcript::{counting_transcript::CountingSponge, TranscriptReader, TranscriptWriter},
        Fr,
    };

    use crate::{claims::Claim, mle::Mle};
    use remainder_shared_types::ff_field;

    use super::*;

    #[test]
    fn test_ligero_input_layer_with_precommit() {
        // Setup phase.
        let layer_id = LayerId::Input(0);
        let rho_inv = 4;
        let ratio = 1.;

        // MLE on 2 variables.
        let evals: Vec<Fr> = [1, 2, 3, 4].into_iter().map(Fr::from).collect();
        let dense_mle = DenseMle::new_from_raw(evals.clone(), layer_id);

        let claim_point = vec![Fr::ONE, Fr::ZERO];
        let claim_result = Fr::from(2);
        let claim: Claim<Fr> = Claim::new(claim_point, claim_result);

        let aux = LigeroAuxInfo::new(evals.len().next_power_of_two(), rho_inv, ratio, None);
        let (pre_commitment, _root) = remainder_ligero_commit(&evals, &aux);

        let verifier_ligero_input_layer =
            LigeroInputLayerDescription::new(layer_id, dense_mle.num_free_vars(), aux);
        let mut ligero_input_layer = LigeroInputLayer::new(
            dense_mle.mle,
            layer_id,
            Some(pre_commitment),
            rho_inv,
            ratio,
        );

        // Transcript writer with test sponge that always returns `1`.
        let mut transcript_writer: TranscriptWriter<Fr, CountingSponge<Fr>> =
            TranscriptWriter::new("Test Transcript Writer");

        // Prover phase.
        // 1. Commit to the input layer.
        let commitment = ligero_input_layer.commit().unwrap();

        // 2. Add commitment to transcript.
        LigeroInputLayer::<Fr>::append_commitment_to_transcript(
            &commitment,
            &mut transcript_writer,
        );

        // 3. ... [skip] proving other layers ...

        // 4. Open commitment (no-op for Public Layers).
        ligero_input_layer
            .open(&mut transcript_writer, claim.clone())
            .unwrap();

        // Verifier phase.
        // 1. Retrieve proof/transcript.
        let transcript = transcript_writer.get_transcript();
        let mut transcript_reader: TranscriptReader<Fr, CountingSponge<Fr>> =
            TranscriptReader::new(transcript);

        // 2. Get commitment from transcript.
        let commitment = verifier_ligero_input_layer
            .get_commitment_from_transcript(&mut transcript_reader)
            .unwrap();

        // 3. ... [skip] verify other layers.

        // 4. Verify this layer's commitment.
        verifier_ligero_input_layer
            .verify(&commitment, claim, &mut transcript_reader)
            .unwrap();
    }

    #[test]
    fn test_ligero_input_layer_no_precommit() {
        // Setup phase.
        let layer_id = LayerId::Input(0);
        let rho_inv = 4;
        let ratio = 1.;

        // MLE on 2 variables.
        let evals: Vec<Fr> = [1, 2, 3, 4].into_iter().map(Fr::from).collect();
        let dense_mle = DenseMle::new_from_raw(evals.clone(), layer_id);

        let claim_point = vec![Fr::ONE, Fr::ZERO];
        let claim_result = Fr::from(2);
        let claim: Claim<Fr> = Claim::new(claim_point, claim_result);

        let verifier_ligero_input_layer = LigeroInputLayerDescription::new(
            layer_id,
            dense_mle.num_free_vars(),
            LigeroAuxInfo::new(evals.len(), rho_inv, ratio, None),
        );
        let mut ligero_input_layer =
            LigeroInputLayer::new(dense_mle.mle, layer_id, None, rho_inv, ratio);

        // Transcript writer with test sponge that always returns `1`.
        let mut transcript_writer: TranscriptWriter<Fr, CountingSponge<Fr>> =
            TranscriptWriter::new("Test Transcript Writer");

        // Prover phase.
        // 1. Commit to the input layer.
        let commitment = ligero_input_layer.commit().unwrap();

        // 2. Add commitment to transcript.
        LigeroInputLayer::<Fr>::append_commitment_to_transcript(
            &commitment,
            &mut transcript_writer,
        );

        // 3. ... [skip] proving other layers ...

        // 4. Open commitment (no-op for Public Layers).
        ligero_input_layer
            .open(&mut transcript_writer, claim.clone())
            .unwrap();

        // Verifier phase.
        // 1. Retrieve proof/transcript.
        let transcript = transcript_writer.get_transcript();
        let mut transcript_reader: TranscriptReader<Fr, CountingSponge<Fr>> =
            TranscriptReader::new(transcript);

        // 2. Get commitment from transcript.
        let commitment = verifier_ligero_input_layer
            .get_commitment_from_transcript(&mut transcript_reader)
            .unwrap();

        // 3. ... [skip] verify other layers.

        // 4. Verify this layer's commitment.
        verifier_ligero_input_layer
            .verify(&commitment, claim, &mut transcript_reader)
            .unwrap();
    }
}

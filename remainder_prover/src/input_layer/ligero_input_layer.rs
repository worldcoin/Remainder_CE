//! An InputLayer that will have it's claim proven with a Ligero Opening Proof.

use remainder_ligero::{
    adapter::{convert_halo_to_lcpc, LigeroProof},
    ligero_commit::{
        remainder_ligero_commit, remainder_ligero_eval_prove, remainder_ligero_verify,
    },
    ligero_structs::LigeroEncoding,
    poseidon_ligero::PoseidonSpongeHasher,
    LcCommit, LcProofAuxiliaryInfo, LcRoot,
};
use remainder_shared_types::{
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::wlx_eval::YieldWLXEvals,
    layer::LayerId,
    mle::{dense::DenseMle, mle_enum::MleEnum},
};

use super::{get_wlx_evaluations_helper, InputLayer, InputLayerError, MleInputLayer};

/// An input layer in which `mle` will be committed to using the Ligero polynomial
/// commitment scheme.
pub struct LigeroInputLayer<F: FieldExt> {
    /// The MLE which we wish to commit to.
    pub mle: DenseMle<F,>,
    /// The ID corresponding to this layer.
    pub(crate) layer_id: LayerId,
    /// The Ligero commitment to `mle`.
    comm: Option<LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>>,
    /// The auxiliary information needed in order to perform an opening proof.
    aux: Option<LcProofAuxiliaryInfo>,
    /// The Merkle root corresponding to the commitment.
    root: Option<LcRoot<LigeroEncoding<F>, F>>,
    /// Whether this layer has already been committed to.
    is_precommit: bool,
    /// The rho inverse for the Reed Solomon encoding.
    rho_inv: Option<u8>,
    /// The ratio of the number of rows : number of columns of the matrix.
    ratio: Option<f64>,
}

/// The Ligero evaluation proof that the prover needs to send to the verifier.
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LigeroInputProof<F: FieldExt> {
    /// The proof itself, see [LigeroProof].
    pub proof: LigeroProof<F>,
    /// The auxiliary information needed to verify the above proof.
    pub aux: LcProofAuxiliaryInfo,
    /// Whether this is a pre-committed (true) or live-committed Ligero input layer
    pub is_precommit: bool,
}

/// The Ligero commitment the prover needs to send to the verifier
pub type LigeroCommitment<F> = LcRoot<LigeroEncoding<F>, F>;

impl<F: FieldExt> InputLayer<F> for LigeroInputLayer<F> {
    type Commitment = LigeroCommitment<F>;

    type OpeningProof = LigeroInputProof<F>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        // --- If we've already generated a commitment (i.e. through `new_with_ligero_commitment()`), ---
        // --- no need to regenerate the commitment ---
        if let (Some(_), Some(_), Some(root)) = (&self.comm, &self.aux, &self.root) {
            return Ok(root.clone());
        }

        let (_, _comm, _root, _aux) =
            remainder_ligero_commit(&self.mle.mle, self.rho_inv.unwrap(), self.ratio.unwrap());
        let (_, comm, root, aux) =
            remainder_ligero_commit(&self.mle.mle, self.rho_inv.unwrap(), self.ratio.unwrap());

        self.comm = Some(comm);
        self.aux = Some(aux);
        self.root = Some(root.clone());

        Ok(root)
    }

    /// Add the commitment to the prover transcript for Fiat-Shamir.
    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        transcript_writer.append("Ligero Merkle Commitment", commitment.clone().into_raw());
    }

    /// Add the commitment to the verifier transcript for Fiat-Shamir.
    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError> {
        let transcript_commitment = transcript_reader
            .consume_element("Ligero Merkle Commitment")
            .map_err(InputLayerError::TranscriptError)?;
        debug_assert_eq!(transcript_commitment, commitment.clone().into_raw());
        Ok(())
    }

    /// "Open" the commitment, in other words, see whether the polynomial evaluated at the
    /// random point in `claim` corresopnds to the claimed value in `claim` by "opening"
    /// the commitment and this random point.
    fn open(
        &self,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
        claim: crate::claims::Claim<F>,
    ) -> Result<Self::OpeningProof, InputLayerError> {
        let aux = self
            .aux
            .clone()
            .ok_or(InputLayerError::OpeningBeforeCommitment)?;
        let comm = self
            .comm
            .clone()
            .ok_or(InputLayerError::OpeningBeforeCommitment)?;
        let root = self
            .root
            .clone()
            .ok_or(InputLayerError::OpeningBeforeCommitment)?;

        let ligero_eval_proof: LigeroProof<F> = remainder_ligero_eval_prove(
            &self.mle.mle,
            claim.get_point(),
            transcript_writer,
            aux.clone(),
            comm,
            root,
        );

        Ok(LigeroInputProof {
            proof: ligero_eval_proof,
            aux,
            is_precommit: self.is_precommit,
        })
    }

    /// Verify the evaluation proof generated from the `open()` function.
    fn verify(
        _commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: crate::claims::Claim<F>,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), super::InputLayerError> {
        let ligero_aux = &opening_proof.aux;
        let (_, ligero_eval_proof, _) =
            convert_halo_to_lcpc(opening_proof.aux.clone(), opening_proof.proof.clone());
        remainder_ligero_verify::<F, _>(
            &ligero_eval_proof,
            ligero_aux.clone(),
            transcript_reader,
            claim.get_point(),
            claim.get_result(),
        );
        Ok(())
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F,> {
        self.mle.clone()
    }
}

impl<F: FieldExt> MleInputLayer<F> for LigeroInputLayer<F> {
    fn new(mle: DenseMle<F,>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            comm: None,
            aux: None,
            root: None,
            is_precommit: false,
            rho_inv: None,
            ratio: None,
        }
    }
}

impl<F: FieldExt> LigeroInputLayer<F> {
    /// Creates new Ligero input layer WITH a precomputed Ligero commitment
    pub fn new_with_ligero_commitment(
        mle: DenseMle<F,>,
        layer_id: LayerId,
        ligero_comm: LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
        ligero_aux: LcProofAuxiliaryInfo,
        ligero_root: LcRoot<LigeroEncoding<F>, F>,
        verifier_is_precommit: bool,
    ) -> Self {
        Self {
            mle,
            layer_id,
            comm: Some(ligero_comm),
            aux: Some(ligero_aux),
            root: Some(ligero_root),
            is_precommit: verifier_is_precommit,
            rho_inv: None,
            ratio: None,
        }
    }

    /// Creates new Ligero input layer with specified rho inverse
    pub fn new_with_rho_inv_ratio(
        mle: DenseMle<F,>,
        layer_id: LayerId,
        rho_inv: u8,
        ratio: f64,
    ) -> Self {
        Self {
            mle,
            layer_id,
            comm: None,
            aux: None,
            root: None,
            is_precommit: false,
            rho_inv: Some(rho_inv),
            ratio: Some(ratio),
        }
    }
}

impl<F: FieldExt> YieldWLXEvals<F> for LigeroInputLayer<F> {
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
            self,
            claim_vecs,
            claimed_vals,
            claimed_mles,
            num_claims,
            num_idx,
        )
    }
}

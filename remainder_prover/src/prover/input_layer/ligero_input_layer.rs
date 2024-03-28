//! An InputLayer that will be have it's claim proven with a Ligero Opening Proof

use std::marker::PhantomData;

use remainder_ligero::{
    adapter::{convert_halo_to_lcpc, LigeroProof},
    ligero_commit::{
        remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify,
    },
    ligero_structs::LigeroEncoding,
    poseidon_ligero::PoseidonSpongeHasher,
    LcCommit, LcProofAuxiliaryInfo, LcRoot,
};
use remainder_shared_types::{
    transcript::{Transcript, TranscriptError},
    FieldExt,
};
use serde::{Deserialize, Serialize};

use crate::{layer::LayerId, mle::dense::DenseMle, prover::input_layer::InputLayerError};

use super::{enum_input_layer::InputLayerEnum, InputLayer, MleInputLayer};

pub struct LigeroInputLayer<F: FieldExt> {
    pub mle: DenseMle<F, F>,
    pub(crate) layer_id: LayerId,
    comm: Option<LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>>,
    aux: Option<LcProofAuxiliaryInfo>,
    root: Option<LcRoot<LigeroEncoding<F>, F>>,
    is_precommit: bool,
    rho_inv: Option<u8>,
    ratio: Option<f64>,
}

/// The *actual* Ligero evaluation proof the prover needs to send to the verifier
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LigeroInputProof<F: FieldExt> {
    pub proof: LigeroProof<F>,
    pub aux: LcProofAuxiliaryInfo,
    /// Whether this is a pre-committed (true) or live-committed Ligero input layer
    pub is_precommit: bool,
}

const RHO_INV: u8 = 4;

/// The *actual* Ligero commitment the prover needs to send to the verifier
pub type LigeroCommitment<F> = LcRoot<LigeroEncoding<F>, F>;

impl<F: FieldExt> InputLayer<F> for LigeroInputLayer<F> {
    type Commitment = LigeroCommitment<F>;

    type OpeningProof = LigeroInputProof<F>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        // --- If we've already generated a commitment (i.e. through `new_with_ligero_commitment()`), ---
        // --- no need to regenerate the commitment ---
        match (&self.comm, &self.aux, &self.root) {
            (Some(_), Some(_), Some(root)) => {
                return Ok(root.clone());
            }
            _ => {}
        }

        let (_, comm, root, aux) = remainder_ligero_commit_prove(
            &self.mle.mle,
            self.rho_inv.unwrap(),
            self.ratio.unwrap(),
        );

        self.comm = Some(comm);
        self.aux = Some(aux);
        self.root = Some(root.clone());

        Ok(root)
    }

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut impl Transcript<F>,
    ) -> Result<(), TranscriptError> {
        transcript.append_field_element("Ligero Merkle Commitment", commitment.clone().into_raw())
    }

    fn open(
        &self,
        transcript: &mut impl Transcript<F>,
        claim: crate::layer::claims::Claim<F>,
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
            transcript,
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

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: crate::layer::claims::Claim<F>,
        transcript: &mut impl Transcript<F>,
    ) -> Result<(), super::InputLayerError> {
        let ligero_aux = &opening_proof.aux;
        let (_, ligero_eval_proof, _) =
            convert_halo_to_lcpc(opening_proof.aux.clone(), opening_proof.proof.clone());
        remainder_ligero_verify::<F>(
            commitment,
            &ligero_eval_proof,
            ligero_aux.clone(),
            transcript,
            claim.get_point(),
            claim.get_result(),
        );
        Ok(())
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        self.mle.clone()
    }
}

impl<F: FieldExt> MleInputLayer<F> for LigeroInputLayer<F> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
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
        mle: DenseMle<F, F>,
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
        mle: DenseMle<F, F>,
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

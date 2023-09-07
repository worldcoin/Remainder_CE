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

use crate::{
    layer::LayerId, mle::dense::DenseMle, prover::input_layer::InputLayerError,
    utils::pad_to_nearest_power_of_two,
};

use super::{enum_input_layer::InputLayerEnum, InputLayer, MleInputLayer};

pub struct LigeroInputLayer<F: FieldExt, Tr> {
    mle: DenseMle<F, F>,
    layer_id: LayerId,
    comm: Option<LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>>,
    aux: Option<LcProofAuxiliaryInfo>,
    root: Option<LcRoot<LigeroEncoding<F>, F>>,
    _marker: PhantomData<Tr>,
}

/// The *actual* Ligero evaluation proof the prover needs to send to the verifier
#[derive(Serialize, Deserialize)]
#[serde(bound = "F: FieldExt")]
pub struct LigeroInputProof<F: FieldExt> {
    pub proof: LigeroProof<F>,
    pub aux: LcProofAuxiliaryInfo,
}

const RHO_INV: u8 = 4;

/// The *actual* Ligero commitment the prover needs to send to the verifier
pub type LigeroCommitment<F> = LcRoot<LigeroEncoding<F>, F>;

impl<F: FieldExt, Tr: Transcript<F>> InputLayer<F> for LigeroInputLayer<F, Tr> {
    type Transcript = Tr;

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

        let (_, comm, root, aux) = remainder_ligero_commit_prove(&self.mle.mle, RHO_INV);

        self.comm = Some(comm);
        self.aux = Some(aux);
        self.root = Some(root.clone());

        Ok(root)
    }

    fn append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript: &mut Self::Transcript,
    ) -> Result<(), TranscriptError> {
        transcript.append_field_element("Ligero Merkle Commitment", commitment.clone().into_raw())
    }

    fn open(
        &self,
        transcript: &mut Self::Transcript,
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
            &claim.0,
            transcript,
            aux.clone(),
            comm,
            root,
        );

        Ok(LigeroInputProof {
            proof: ligero_eval_proof,
            aux,
        })
    }

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: crate::layer::claims::Claim<F>,
        transcript: &mut Self::Transcript,
    ) -> Result<(), super::InputLayerError> {
        let ligero_aux = &opening_proof.aux;
        let (_, ligero_eval_proof, _) =
            convert_halo_to_lcpc(opening_proof.aux.clone(), opening_proof.proof.clone());
        remainder_ligero_verify::<F>(
            commitment,
            &ligero_eval_proof,
            ligero_aux.clone(),
            transcript,
            &claim.0,
            claim.1,
        );
        Ok(())
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }

    fn get_padded_mle(&self) -> DenseMle<F, F> {
        self.mle.clone()
    }

    fn to_enum(self) -> InputLayerEnum<F, Self::Transcript> {
        InputLayerEnum::LigeroInputLayer(self)
    }
}

impl<F: FieldExt, Tr: Transcript<F>> MleInputLayer<F> for LigeroInputLayer<F, Tr> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            comm: None,
            aux: None,
            root: None,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, Tr: Transcript<F>> LigeroInputLayer<F, Tr> {
    /// Creates new Ligero input layer WITH a precomputed Ligero commitment
    pub fn new_with_ligero_commitment(
        mle: DenseMle<F, F>,
        layer_id: LayerId,
        ligero_comm: LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>,
        ligero_aux: LcProofAuxiliaryInfo,
        ligero_root: LcRoot<LigeroEncoding<F>, F>,
    ) -> Self {
        Self {
            mle,
            layer_id,
            comm: Some(ligero_comm),
            aux: Some(ligero_aux),
            root: Some(ligero_root),
            _marker: PhantomData,
        }
    }
}

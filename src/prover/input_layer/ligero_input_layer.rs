//! An InputLayer that will be have it's claim proven with a Ligero Opening Proof

use std::marker::PhantomData;

use lcpc_2d::{FieldExt, fs_transcript::halo2_remainder_transcript::{Transcript, TranscriptError}, LcCommit, poseidon_ligero::PoseidonSpongeHasher, ligero_structs::LigeroEncoding, adapter::{LigeroProof, convert_halo_to_lcpc}, ScalarField, ligero_commit::{remainder_ligero_commit_prove, remainder_ligero_eval_prove, remainder_ligero_verify}, LcRoot, LcProofAuxiliaryInfo};

use crate::{mle::dense::DenseMle, layer::LayerId, utils::pad_to_nearest_power_of_two, prover::input_layer::InputLayerError};

use super::{InputLayer, MleInputLayer};

pub struct LigeroInputLayer<F: FieldExt, F2, Tr> {
    mle: DenseMle<F, F>,
    layer_id: LayerId,
    comm: Option<LcCommit<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>>,
    aux: Option<LcProofAuxiliaryInfo>,
    root: Option<LcRoot<LigeroEncoding<F>, F>>,
    _marker: PhantomData<(F2, Tr)>,
}

pub struct LigeroInputProof<F: ScalarField> {
    proof: LigeroProof<F>,
    aux: LcProofAuxiliaryInfo,
}

const RHO_INV: u8 = 4;

pub type LigeroCommitment<F> = LcRoot<LigeroEncoding<F>, F>;

impl<F: FieldExt, F2: ScalarField, Tr: Transcript<F>> InputLayer<F> for LigeroInputLayer<F, F2, Tr> {
    type Transcript = Tr;

    type Commitment = LigeroCommitment<F>;

    type OpeningProof = LigeroInputProof<F2>;

    fn commit(&mut self) -> Result<Self::Commitment, super::InputLayerError> {
        // let orig_input_layer_bookkeeping_table: Vec<F> = pad_to_nearest_power_of_two(self.mle.mle.clone());
        let (_, comm, root, aux) = remainder_ligero_commit_prove(&self.mle.mle, RHO_INV);

        self.comm = Some(comm);
        self.aux = Some(aux);
        self.root = Some(root.clone());

        Ok(root)
    }

    fn append_commitment_to_transcript(commitment: &Self::Commitment, transcript: &mut Self::Transcript) -> Result<(), TranscriptError> {
        transcript.append_field_element("Ligero Merkle Commitment", commitment.clone().into_raw())
    }

    fn open(&self, transcript: &mut Self::Transcript, claim: crate::layer::Claim<F>) -> Result<Self::OpeningProof, InputLayerError> {
        let aux = self.aux.clone().ok_or(InputLayerError::OpeningBeforeCommitment)?;
        let comm = self.comm.clone().ok_or(InputLayerError::OpeningBeforeCommitment)?;
        let root = self.root.clone().ok_or(InputLayerError::OpeningBeforeCommitment)?;

        let ligero_eval_proof: LigeroProof<F2> = remainder_ligero_eval_prove(
            &self.mle.mle,
            &claim.0,
            transcript,
            aux.clone(),
            comm,
            root
        );

        Ok(LigeroInputProof {
            proof: ligero_eval_proof,
            aux
        })
    }

    fn verify(commitment: &Self::Commitment, opening_proof: &Self::OpeningProof, claim: crate::layer::Claim<F>, transcript: &mut Self::Transcript) -> Result<(), super::InputLayerError> {
        let ligero_aux = &opening_proof.aux;
        let (_, ligero_eval_proof, _) = convert_halo_to_lcpc::<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F, F2>(ligero_aux.clone(), opening_proof.proof.clone());
        remainder_ligero_verify::<F, F2>(commitment, &ligero_eval_proof, ligero_aux.clone(), transcript, &claim.0, claim.1);
        Ok(())
    }

    fn layer_id(&self) -> &LayerId {
        &self.layer_id
    }
}

impl<F: FieldExt, F2: ScalarField, Tr: Transcript<F>> MleInputLayer<F> for LigeroInputLayer<F, F2, Tr> {
    fn new(mle: DenseMle<F, F>, layer_id: LayerId) -> Self {
        Self {
            mle,
            layer_id,
            comm: None,
            aux: None,
            root: None,
            _marker: PhantomData
        }
    }
}
//! An InputLayer that will be have it's claim proven with a Ligero Opening Proof

use std::marker::PhantomData;

use ark_std::{cfg_into_iter, end_timer, start_timer};
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
    transcript::{TranscriptReader, TranscriptSponge, TranscriptWriter},
    FieldExt,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use rayon::prelude::{ParallelIterator, IntoParallelIterator};

use crate::{claims::wlx_eval::{get_num_wlx_evaluations, YieldWLXEvals, ENABLE_PRE_FIX}, layer::LayerId, mle::{dense::DenseMle, mle_enum::MleEnum, MleIndex, MleRef}, prover::input_layer::InputLayerError, sumcheck::evaluate_at_a_point};

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

    fn prover_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_writer: &mut TranscriptWriter<F, impl TranscriptSponge<F>>,
    ) {
        transcript_writer.append("Ligero Merkle Commitment", commitment.clone().into_raw());
    }

    fn verifier_append_commitment_to_transcript(
        commitment: &Self::Commitment,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), InputLayerError> {
        let transcript_commitment = transcript_reader
            .consume_element("Ligero Merkle Commitment")
            .map_err(|e| InputLayerError::TranscriptError(e))?;
        debug_assert_eq!(transcript_commitment, commitment.clone().into_raw());
        Ok(())
    }

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

    fn verify(
        commitment: &Self::Commitment,
        opening_proof: &Self::OpeningProof,
        claim: crate::claims::Claim<F>,
        transcript_reader: &mut TranscriptReader<F, impl TranscriptSponge<F>>,
    ) -> Result<(), super::InputLayerError> {
        let ligero_aux = &opening_proof.aux;
        let (_, ligero_eval_proof, _) =
            convert_halo_to_lcpc(opening_proof.aux.clone(), opening_proof.proof.clone());
        remainder_ligero_verify::<F, _>(
            commitment,
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

impl<F: FieldExt> YieldWLXEvals<F> for LigeroInputLayer<F> {
        
    /// Computes the V_d(l(x)) evaluations for the input layer V_d.
    fn get_wlx_evaluations(
        &self,
        claim_vecs: &Vec<Vec<F>>,
        claimed_vals: &Vec<F>,
        claimed_mles: Vec<MleEnum<F>>,
        num_claims: usize,
        num_idx: usize,
    ) -> Result<Vec<F>, crate::claims::ClaimError> {
        let prep_timer = start_timer!(|| "Claim wlx prep");
        let mut mle_ref = self.get_padded_mle().clone().mle_ref();
        end_timer!(prep_timer);
        info!(
            "Wlx MLE len: {}",
            mle_ref.current_mle.get_evals_vector().len()
        );

        //fix variable hella times
        //evaluate expr on the mutated expr

        // get the number of evaluations
        mle_ref.index_mle_indices(0);
        let (num_evals, common_idx) = get_num_wlx_evaluations(claim_vecs);
        let chal_point = &claim_vecs[0];

        if ENABLE_PRE_FIX {
            if common_idx.is_some() {
                let common_idx = common_idx.unwrap();
                common_idx.iter().for_each(|chal_idx| {
                    if let MleIndex::IndexedBit(idx_bit_num) = mle_ref.mle_indices()[*chal_idx] {
                        mle_ref.fix_variable_at_index(idx_bit_num, chal_point[*chal_idx]);
                    }
                });
            }
        }

        debug!("Evaluating {num_evals} times.");

        // we already have the first #claims evaluations, get the next num_evals - #claims evaluations
        let next_evals: Vec<F> = cfg_into_iter!(num_claims..num_evals)
            // let next_evals: Vec<F> = (num_claims..num_evals).into_iter()
            .map(|idx| {
                // get the challenge l(idx)
                let new_chal: Vec<F> = cfg_into_iter!(0..num_idx)
                    // let new_chal: Vec<F> = (0..num_idx).into_iter()
                    .map(|claim_idx| {
                        let evals: Vec<F> = cfg_into_iter!(&claim_vecs)
                            // let evals: Vec<F> = (&claim_vecs).into_iter()
                            .map(|claim| claim[claim_idx])
                            .collect();
                        evaluate_at_a_point(&evals, F::from(idx as u64)).unwrap()
                    })
                    .collect();

                let mut fix_mle = mle_ref.clone();
                {
                    new_chal.into_iter().enumerate().for_each(|(idx, chal)| {
                        if let MleIndex::IndexedBit(idx_num) = fix_mle.mle_indices()[idx] {
                            fix_mle.fix_variable(idx_num, chal);
                        }
                    });
                    fix_mle.current_mle[0]
                }
            })
            .collect();

        // concat this with the first k evaluations from the claims to get num_evals evaluations
        let mut wlx_evals = claimed_vals.clone();
        wlx_evals.extend(&next_evals);
        debug!("Returning evals:\n{:#?} ", wlx_evals);
        Ok(wlx_evals)

    }
}

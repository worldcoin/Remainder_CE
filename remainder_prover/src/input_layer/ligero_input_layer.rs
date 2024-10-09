//! An InputLayer that will have it's claim proven with a Ligero Opening Proof.

use remainder_ligero::{
    ligero_commit::{
        remainder_ligero_commit, remainder_ligero_eval_prove,
    },
    ligero_structs::LigeroAuxInfo,
    poseidon_ligero::PoseidonSpongeHasher,
    LcCommit, LcRoot,
};
use remainder_shared_types::{
    transcript::ProverTranscript,
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    claims::wlx_eval::YieldWLXEvals,
    layer::LayerId,
    mle::{dense::DenseMle, evals::MultilinearExtension},
};

use super::{
    get_wlx_evaluations_helper, InputLayerTrait,
    InputLayerError,
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

impl<F: Field> LigeroInputLayerDescription<F> {
    /// Return the layer id
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
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
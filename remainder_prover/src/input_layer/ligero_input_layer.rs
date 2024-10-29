//! An InputLayer that will have it's claim proven with a Ligero Opening Proof.

use crate::layer::LayerId;
use remainder_ligero::{
    ligero_structs::LigeroAuxInfo, poseidon_ligero::PoseidonSpongeHasher, LcCommit, LcRoot,
};
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

/// The Ligero commitment the prover sees, which contains more information than the verifier should see.
pub type LigeroCommitment<F> = LcCommit<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>;
/// The Ligero commitment the prover sends the verifier (adds to transcript) which is the commitment to the root.
pub type LigeroRoot<F> = LcRoot<LigeroAuxInfo<F>, F>;

/// Type alias for Ligero input layer description + optional precommit,
/// to be used on the prover's end.
pub type LigeroInputLayerDescriptionWithPrecommit<F> = (
    LigeroInputLayerDescription<F>,
    Option<LcCommit<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>>,
);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Hash)]
#[serde(bound = "F: Field")]
/// The circuit description of a [LigeroInputLayer]. Stores the shape information of this layer.
pub struct LigeroInputLayerDescription<F: Field> {
    /// The ID of this Ligero Input Layer.
    pub layer_id: LayerId,

    /// The number of variables this Ligero Input Layer is on.
    pub num_vars: usize,

    /// The auxiliary information needed to verify the proof.
    pub aux: LigeroAuxInfo<F>,
}

//! An InputLayer that will have it's claim proven with a Ligero Opening Proof.

use remainder_ligero::{
    ligero_structs::LigeroAuxInfo,
    poseidon_ligero::PoseidonSpongeHasher,
    LcCommit, LcRoot,
};
use remainder_shared_types::{
    Field,
};
use serde::{Deserialize, Serialize};

use crate::{
    layer::LayerId,
    mle::{evals::MultilinearExtension},
};


/// An input layer in which `mle` will be committed to using the Ligero polynomial
/// commitment scheme.
#[derive(Debug)]
pub struct LigeroInputLayer<F: Field> {
    /// The MLE which we wish to commit to.
    pub mle: MultilinearExtension<F>,
    /// The ID corresponding to this layer.
    pub(crate) layer_id: LayerId,
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

impl<F: Field> LigeroInputLayerDescription<F> {
    /// Return the layer id
    pub fn layer_id(&self) -> LayerId {
        self.layer_id
    }
}
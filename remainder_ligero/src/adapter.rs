use crate::ligero_structs::LigeroRoot;
use crate::poseidon_ligero::poseidon_digest::FieldHashFnDigest;
use crate::{LcColumn, LcEncoding};

use crate::poseidon_ligero::PoseidonSpongeHasher;
use crate::{ligero_structs::LigeroAuxInfo, ligero_structs::LigeroEvalProof};

use itertools::Itertools;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

/// Struct containing all of the components of a Ligero commitment + evaluation
/// proof, as required by the Halo2-GKR verifier.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LigeroProof<F> {
    /// Root of the Merkle tree
    pub merkle_root: F,
    /// List of products v_i.A, where v_i is the tensor constructed from (half of) the i-th opened point
    pub v_0_a: Vec<Vec<F>>,
    /// List of full columns queried by the verifier
    pub columns: Vec<Vec<F>>,
    /// List of Merkle openings
    pub merkle_paths: Vec<Vec<F>>,
    /// List of all column indices to open at (technically redundant but helpful for debugging and back-conversion)
    pub col_indices: Vec<usize>,
}

/// Converts a lcpc-style Ligero proof/root into the above data structure.
pub fn convert_lcpc_to_halo<F: Field>(
    root: LigeroRoot<F>,
    pf: LigeroEvalProof<PoseidonSpongeHasher<F>, LigeroAuxInfo<F>, F>,
) -> LigeroProof<F> {
    let merkle_root = root.root;

    // we convert this into a vector, since the circuit for the Ligero verifier
    // assumes that we can have multiple point openings
    let v_0_a = vec![pf.p_eval];

    let columns: Vec<Vec<F>> = pf
        .columns
        .clone()
        .into_iter()
        .map(|lc_column| lc_column.col)
        .collect();

    let merkle_paths: Vec<Vec<F>> = pf
        .columns
        .clone()
        .into_iter()
        .map(|lc_column| lc_column.path)
        .collect();

    let col_indices: Vec<usize> = pf
        .columns
        .into_iter()
        .map(|lc_column| lc_column.col_idx)
        .collect();

    LigeroProof {
        merkle_root,
        v_0_a,
        columns,
        merkle_paths,
        col_indices,
    }
}

/// Converts the Halo2-compatible proof back into the Ligero structs needed
/// for the `verify()` function.
///
/// ## Arguments
/// * `aux` - Auxiliary proof info (from the prove phase)
/// * `halo2_ligero_proof` - The already-converted Halo2-compatible Ligero proof (also from the prove + convert phase)
///
/// ## Returns
/// * `root` - Ligero commitment root
/// * `ligero_eval_proof` - The evaluation proof (including columns + openings)
/// * `enc` - The encoding (should be deprecated, but haven't had time yet TODO!(ryancao))
pub fn convert_halo_to_lcpc<D, E, F>(
    aux: &LigeroAuxInfo<F>,
    halo2_ligero_proof: LigeroProof<F>,
) -> (LigeroRoot<F>, LigeroEvalProof<D, E, F>, LigeroAuxInfo<F>)
where
    F: Field,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{
    // --- Unpacking the Merkle root ---
    let root = LigeroRoot::<F> {
        root: halo2_ligero_proof.merkle_root,
        _p: std::marker::PhantomData,
    };

    let ligero_eval_proof = LigeroEvalProof::<D, E, F> {
        encoded_num_cols: aux.encoded_num_cols,
        p_eval: halo2_ligero_proof.v_0_a[0].clone(),
        columns: halo2_ligero_proof
            .col_indices
            .into_iter()
            .zip(
                halo2_ligero_proof
                    .columns
                    .into_iter()
                    .zip(halo2_ligero_proof.merkle_paths),
            )
            .map(|(col_idx, (column, merkle_path))| LcColumn::<E, F> {
                col_idx,
                col: column,
                path: merkle_path,
                phantom_data: std::marker::PhantomData,
            })
            .collect_vec(),
        phantom_data: std::marker::PhantomData,
    };

    (root, ligero_eval_proof, aux.clone())
}

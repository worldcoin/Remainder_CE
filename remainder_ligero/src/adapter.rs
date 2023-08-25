use std::marker::PhantomData;

use crate::{LcEncoding, LcColumn};
use crate::poseidon_ligero::poseidon_digest::FieldHashFnDigest;

use crate::poseidon_ligero::PoseidonSpongeHasher;
use crate::LcProofAuxiliaryInfo;
use crate::{ligero_structs::LigeroEncoding, ligero_structs::LigeroEvalProof, LcRoot};
use remainder_shared_types::FieldExt;
use halo2_base::utils::log2_ceil;

use serde::{Serialize, Deserialize};
use itertools::Itertools;

/// Following the spec from Notion!
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LigeroProof<F> {
    /// Root of the Merkle tree
    pub merkle_root: F,
    /// Product r.A, where r is the random vector and A is the coefficient matrix
    pub r_a: Vec<F>,
    /// List of products v_i.A, where v_i is the tensor constructed from (half of) the i-th opened point
    pub v_0_a: Vec<Vec<F>>,
    /// List of full columns queried by the verifier
    pub columns: Vec<Vec<F>>,
    /// List of Merkle openings
    pub merkle_paths: Vec<Vec<F>>,
    /// List of all column indices to open at (technically redundant but helpful for debugging and back-conversion)
    pub col_indices: Vec<usize>
}

/// Complementary information to a LigeroProof necessary
/// in some contexts: point and evaluation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LigeroClaim<F> {
    /// The opened point
    pub point: Vec<F>,
    /// The value of the polynomial at the point
    pub eval: F,
}

/// Converts a lcpc-style Ligero proof/root into the above data structure.
pub fn convert_lcpc_to_halo<F: FieldExt>(
    aux: LcProofAuxiliaryInfo,
    root: LcRoot<LigeroEncoding<F>, F>,
    pf: LigeroEvalProof::<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>
) -> LigeroProof<F> {

    let merkle_root = root.root;

    assert_eq!(pf.p_random_vec.len(), 1);

    let r_a = pf.p_random_vec[0].clone();
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
        .columns.clone()
        .into_iter()
        .map(|lc_column| lc_column.path)
        .collect();

    let col_indices: Vec<usize> = pf
        .columns
        .into_iter()
        .map(|lc_column| lc_column.col_idx)
        .collect();

    // --- Printing the parameters in a Rust-ready format ---
    println!(
        "const LOG_M_TEST: usize = {}",
        log2_ceil(aux.orig_num_cols as u64)
    );
    println!("const RHO_INV_TEST: usize = {}", aux.rho_inv);
    println!(
        "const LOG_N_TEST: usize = {}",
        log2_ceil(aux.num_rows as u64)
    );
    println!("const T_TEST: usize =  {}", columns.len());

    // let claim = LigeroClaim {
    //     point: vec_ark_to_halo(&raw_claim.point),
    //     eval: ark_to_halo(&raw_claim.eval),
    // };
    
    let proof = LigeroProof {
        merkle_root,
        r_a,
        v_0_a,
        columns,
        merkle_paths,
        col_indices
    };

    proof

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
    aux: LcProofAuxiliaryInfo,
    halo2_ligero_proof: LigeroProof<F>
) -> (LcRoot<LigeroEncoding<F>, F>, LigeroEvalProof<D, E, F>, LigeroEncoding<F>)
where
    F: FieldExt,
    D: FieldHashFnDigest<F> + Send + Sync,
    E: LcEncoding<F> + Send + Sync,
{

    // --- Unpacking the Merkle root ---
    let root = LcRoot::<LigeroEncoding<F>, F>{
        root: halo2_ligero_proof.merkle_root,
        _p: std::marker::PhantomData,
    };

    let ligero_eval_proof = LigeroEvalProof::<D, E, F>{
        encoded_num_cols: aux.encoded_num_cols,
        p_eval: halo2_ligero_proof.v_0_a[0].clone(),
        p_random_vec: vec![halo2_ligero_proof.r_a],
        columns: halo2_ligero_proof
            .col_indices
            .into_iter()
            .zip(
                halo2_ligero_proof.columns.into_iter().zip(
                    halo2_ligero_proof.merkle_paths.into_iter()
                )
            )
            .map(|(col_idx, (column, merkle_path))| {
            LcColumn::<E, F>{
                col_idx,
                col: column,
                path: merkle_path,
                phantom_data: std::marker::PhantomData,
            }
        }).collect_vec(),
        phantom_data: std::marker::PhantomData,
    };

    let enc = LigeroEncoding {
        orig_num_cols: aux.orig_num_cols,
        encoded_num_cols: aux.encoded_num_cols,
        phantom: std::marker::PhantomData,
        rho_inv: aux.rho_inv,
    };

    (root, ligero_eval_proof, enc)

}

// /// Converts the saved Ligero proof/root into the above data structure and serializes it to a file ready for Halo2 Ligero verifier
// pub fn load_and_convert<F: FieldExt>(
//     aux_path: &str,
//     proof_path: &str,
//     root_path: &str,
// ) -> LigeroProof<F> {
//     // --- Loads and converts into byte vector ---
//     let encoded_auxiliary_bytes =
//         fs::read(aux_path).expect("Unable to read proof aux info from file");
//     let encoded_proof_bytes = fs::read(proof_path).expect("Unable to read proof from file");
//     let encoded_root_bytes = fs::read(root_path).expect("Unable to read root from file");

//     // --- Loads into the original Ligero data structures ---
//     let aux = LcProofAuxiliaryInfo::deserialize_compressed(&*encoded_auxiliary_bytes).unwrap();
//     let root =
//         LcRoot::<LigeroEncoding<F>, F>::deserialize_compressed(&*encoded_root_bytes).unwrap();
//     let pf =
//         LigeroEvalProof::<PoseidonSpongeHasher<F>, LigeroEncoding<F>, F>::deserialize_compressed(
//             &*encoded_proof_bytes,
//         )
//         .unwrap();

//     // --- Performs conversion on loaded structs ---
//     convert_lcpc_to_halo(aux, root, pf)
// }

#![allow(warnings)]
use remainder::prover::GKRCircuitDescription;
use remainder_shared_types::halo2curves::{bn256::G1 as Bn256Point, group::Group};
use remainder_shared_types::transcript::Transcript;

use crate::hyrax_gkr::HyraxProof;

pub enum UpgradeError {}

type Scalar = <Bn256Point as Group>::Scalar;
/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives the image, commitment and
/// blinding for each combination of version, eye, and type (iris or mask), and returns, for each
/// such combination, the transcript and the corresponding iris/mask code (in the clear, as bytes).
pub fn prove_upgrade_to_v3(
    v2_left_image: &[u8],             // v2, left eye, image
    v2_left_image_commitment: &[u8],  // commitment for the same
    v2_left_image_blinding: &[u8],    // blinding for the same
    v2_left_mask: &[u8],              // v2, left eye, mask
    v2_left_mask_commitment: &[u8],   // commitment for the same
    v2_left_mask_blinding: &[u8],     // blinding for the same
    v2_right_image: &[u8],            // v2, right eye, image
    v2_right_image_commitment: &[u8], // commitment for the same
    v2_right_image_blinding: &[u8],   // blinding for the same
    v2_right_mask: &[u8],             // v2, right eye, mask
    v2_right_mask_commitment: &[u8],  // commitment for the same
    v2_right_mask_blinding: &[u8],    // blinding for the same
    v3_left_image: &[u8],             // v3, left eye, image
    v3_left_image_commitment: &[u8],  // commitment for the same
    v3_left_image_blinding: &[u8],    // blinding for the same
    v3_left_mask: &[u8],              // v3, left eye, mask
    v3_left_mask_commitment: &[u8],   // commitment for the same
    v3_left_mask_blinding: &[u8],     // blinding for the same
    v3_right_image: &[u8],            // v3, right eye, image
    v3_right_image_commitment: &[u8], // commitment for the same
    v3_right_image_blinding: &[u8],   // blinding for the same
    v3_right_mask: &[u8],             // v3, right eye, mask
    v3_right_mask_commitment: &[u8],  // commitment for the same
    v3_right_mask_blinding: &[u8],    // blinding for the same
) -> Result<
    (
        (HyraxProof<Bn256Point>, Vec<u8>), // v2, left eye, image
        (HyraxProof<Bn256Point>, Vec<u8>), // v2, left eye, mask
        (HyraxProof<Bn256Point>, Vec<u8>), // v2, right eye, image
        (HyraxProof<Bn256Point>, Vec<u8>), // ...
        (HyraxProof<Bn256Point>, Vec<u8>),
        (HyraxProof<Bn256Point>, Vec<u8>),
        (HyraxProof<Bn256Point>, Vec<u8>),
        (HyraxProof<Bn256Point>, Vec<u8>), // v3, right eye, mask
    ),
    UpgradeError,
> {
    /*
    will need to fetch the Pedersen committer from somewhere, or regenerate it from a string.
    There will be a helper fn:
        creates a new transcript each time.
        accepts the blinding_rng and the converter and the pedersen committer.
        accepts a prover commitment to the image
        accepts a prover commitment to the iris/mask code
        calls something like `circuit_description_and_inputs`
        returns the proof and a commitment to the iris/mask

    the calling context (of the helper) will create an iris/mask code commitment in the v3 case, and re-use an existing one in the v2 case.
    it will need to return the full prover commitment in the v3 case, since these would be needed e.g. to prove a v3 to v4 upgrade.

    for each version:
        for each of [iris, mask]:
            select the appropriate kernel values and thresholds (these are all constants available in the source code)
            fetch the pre-loaded transcript for this version and type (iris vs mask)
            for each eye:
                derive the iris/mask code using the clear-text image, kernel values and thresholds (use this as a _public_ input).
                append image commitment and iris/mask code to transcript.
                create the proof that the iris/mask code corresponds to the image and the kernel values.
    */
    todo!()
}

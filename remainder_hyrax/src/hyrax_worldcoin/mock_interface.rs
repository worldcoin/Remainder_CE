use remainder_shared_types::transcript::Transcript;
use crate::prover::GKRCircuitDescription;

type Scalar = <Bn256Point as Group>::Scalar;
/// Prove the upgrade from v2 to v3 using the Hyrax proof system. Receives the image, commitment and
/// blinding for each combination of version, eye, and type (iris or mask), and returns, for each
/// such combination, the transcript and commitment information for corresponding iris/mask code.
/// Each commitment is a tuple of the iris/mask code in plain text, the commitment itself, and the
/// blinding factors.
pub fn prove_upgrade_to_v3(
    v2_left_image: &[u8],  // v2, left eye, image
    v2_left_image_commitment: &[u8], // commitment for the same
    v2_left_image_blinding: &[u8],  // blinding for the same
    v2_left_mask: &[u8],  // v2, left eye, mask
    v2_left_mask_commitment: &[u8], // commitment for the same
    v2_left_mask_blinding: &[u8],  // blinding for the same
    v2_right_image: &[u8],  // v2, right eye, image
    v2_right_image_commitment: &[u8], // commitment for the same
    v2_right_image_blinding: &[u8],  // blinding for the same
    v2_right_mask: &[u8],  // v2, right eye, mask
    v2_right_mask_commitment: &[u8], // commitment for the same
    v2_right_mask_blinding: &[u8],  // blinding for the same
    v3_left_image: &[u8],  // v3, left eye, image
    v3_left_image_commitment: &[u8], // commitment for the same
    v3_left_image_blinding: &[u8],  // blinding for the same
    v3_left_mask: &[u8],  // v3, left eye, mask
    v3_left_mask_commitment: &[u8], // commitment for the same
    v3_left_mask_blinding: &[u8],  // blinding for the same
    v3_right_image: &[u8],  // v3, right eye, image
    v3_right_image_commitment: &[u8], // commitment for the same
    v3_right_image_blinding: &[u8],  // blinding for the same
    v3_right_mask: &[u8],  // v3, right eye, mask
    v3_right_mask_commitment: &[u8], // commitment for the same
    v3_right_mask_blinding: &[u8],  // blinding for the same
) -> Result<(
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)), // v2, left eye, image
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)), // v2, left eye, mask
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)), // v2, right eye, image
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)), // ...
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)),
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)),
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>)),
        (Transcript<Scalar>, (Vec<u8>, Vec<u8>, Vec<u8>))  // v3, right eye, mask
    ), UpgradeError> {
    /*
    for each version:
        for each of [iris, mask]:
            select the appropriate kernel values and thresholds (these are all constants available in the source code)
            fetch the pre-loaded transcript for this version and type (iris vs mask)
            for each eye:
                derive the iris/mask code using the clear-text image, kernel values and thresholds.
                create a commitment to the iris/mask code.
                append image commitment and iris/mask code commitment to transcript.
                create the proof that the iris/mask code corresponds to the image and the kernel values.
    */
    todo!()
}
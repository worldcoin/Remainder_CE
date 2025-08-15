//! Hyrax IP wrapper over Remainder's GKR prover.

// Note that this allows us to pass in `&mut rng` to various functions
// which needlessly take ownership (unfortunately, these functions are
// not part of Remainder and are therefore out of our control) over the
// `impl Rng` which is passed in.
#![allow(clippy::needless_borrows_for_generic_args)]
pub mod hyrax_gkr;
pub mod hyrax_pcs;
pub mod hyrax_primitives;
pub mod utils;

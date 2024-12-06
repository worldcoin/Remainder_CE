use once_cell::sync::Lazy;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::circuit_hash::CircuitHashType;

use super::{
    ClaimAggregationStrategy, GKRCircuitProverConfig, GKRCircuitVerifierConfig, ProofConfig,
};

/// Default prover/verifier config is runtime-optimized for both! Note that this
/// is a global static variable with atomic read/write access. In particular:
/// * The [Lazy] component allows there to be a single static global config.
///     This ensures that all prover/verifier functionality is "viewing" the
///     same global config and we don't need to pass the options around.
/// * The [RwLock] component ensures that there is atomic access to the prover
///     and verifier configs. This ensures that we are able to atomically (with
///     respect to the config) check that the global static config matches the
///     locally expected config (e.g. one which you might specify for a specific
///     test or binary run) and run the proving/verifying functions under that
///     global config with the guarantee that the global config remains the same
///     (since we simply hold onto a reader for the lifetime of the function
///     call).
/// * Finally, the idea of having the [GKRCircuitProverConfig] and
///     [GKRCircuitVerifierConfig] together in the same [RwLock] is so that
///     we can run an atomic prove + verify test. This is technically less
///     efficient vs. keeping each of the configs in independent [Lazy<RwLock>],
///     but gives a nicer abstraction.
pub static PROVER_VERIFIER_CONFIG: Lazy<
    RwLock<(GKRCircuitProverConfig, GKRCircuitVerifierConfig)>,
> = Lazy::new(|| {
    let default_prover_config = GKRCircuitProverConfig::runtime_optimized_default();
    let default_verifier_config =
        GKRCircuitVerifierConfig::new_from_prover_config(&default_prover_config, false);
    RwLock::new((default_prover_config, default_verifier_config))
});

// -------------------- Helper functions for setting global config --------------------

/// Sets the global prover config to be equivalent to the expected config passed in.
pub fn set_global_prover_config(
    expected_prover_config: &GKRCircuitProverConfig,
    prover_verifier_config_instance: &mut RwLockWriteGuard<
        '_,
        (GKRCircuitProverConfig, GKRCircuitVerifierConfig),
    >,
) {
    prover_verifier_config_instance.0 = expected_prover_config.clone();
}

/// Sets the global verifier config to be equivalent to the expected config passed in.
pub fn set_global_verifier_config(
    expected_verifier_config: &GKRCircuitVerifierConfig,
    prover_verifier_config_instance: &mut RwLockWriteGuard<
        '_,
        (GKRCircuitProverConfig, GKRCircuitVerifierConfig),
    >,
) {
    prover_verifier_config_instance.1 = *expected_verifier_config;
}

// -------------------- Helper functions for accessing global fields (prover) --------------------

/// Whether to turn on "lazy beta evals" optimization
/// (see documentation within [GKRCircuitProverConfig])
pub fn global_prover_lazy_beta_evals() -> bool {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance.0.get_lazy_beta_evals()
}

/// Circuit description to be hashed and sent to verifier within transcript
/// (see documentation within [GKRCircuitProverConfig])
pub fn global_prover_circuit_description_hash_type() -> CircuitHashType {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance
        .0
        .get_circuit_description_hash_type()
}
/// The variant of claim aggregation to be used
/// (see documentation within [GKRCircuitProverConfig])
pub fn global_prover_claim_agg_strategy() -> ClaimAggregationStrategy {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance.0.get_claim_agg_strategy()
}
/// Whether to turn on "constant columns" optimization for deterministic claim agg
/// (see documentation within [GKRCircuitProverConfig])
pub fn global_prover_claim_agg_constant_column_optimization() -> bool {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance
        .0
        .get_claim_agg_constant_column_optimization()
}
/// Whether to turn on the "bit packed vector" optimization
/// (see documentation within [GKRCircuitProverConfig])
pub fn global_prover_enable_bit_packing() -> bool {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance.0.get_enable_bit_packing()
}

/// Returns a copy of the global [GKRCircuitProverConfig].
pub fn get_current_global_prover_config() -> GKRCircuitProverConfig {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    GKRCircuitProverConfig::new(
        prover_verifier_config_instance.0.get_lazy_beta_evals(),
        prover_verifier_config_instance
            .0
            .get_circuit_description_hash_type(),
        prover_verifier_config_instance.0.get_claim_agg_strategy(),
        prover_verifier_config_instance
            .0
            .get_claim_agg_constant_column_optimization(),
        prover_verifier_config_instance.0.get_enable_bit_packing(),
    )
}

// -------------------- Helper functions for accessing global fields (verifier) --------------------

/// Whether to turn on the "lazy beta evals" optimization
/// (see documentation within [GKRCircuitVerifierConfig])
pub fn global_verifier_lazy_beta_evals() -> bool {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance.1.get_lazy_beta_evals()
}
/// The type of hash function to use for hashing the circuit into the transcript
/// (see documentation within [GKRCircuitVerifierConfig])
pub fn global_verifier_circuit_description_hash_type() -> CircuitHashType {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance
        .1
        .get_circuit_description_hash_type()
}
/// Whether to use RLC or deterministic claim agg for verification
/// (see documentation within [GKRCircuitVerifierConfig])
pub fn global_verifier_claim_agg_strategy() -> ClaimAggregationStrategy {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance.1.get_claim_agg_strategy()
}
/// Whether to turn on the "constant columns" optimization for claim agg.
/// (see documentation within [GKRCircuitVerifierConfig])
pub fn global_verifier_claim_agg_constant_column_optimization() -> bool {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    prover_verifier_config_instance
        .1
        .get_claim_agg_constant_column_optimization()
}

/// Returns a copy of the global [GKRCircuitVerifierConfig].
pub fn get_current_global_verifier_config() -> GKRCircuitVerifierConfig {
    let prover_verifier_config_instance = PROVER_VERIFIER_CONFIG.read();
    GKRCircuitVerifierConfig::new(
        prover_verifier_config_instance.1.get_lazy_beta_evals(),
        prover_verifier_config_instance
            .1
            .get_circuit_description_hash_type(),
        prover_verifier_config_instance.1.get_claim_agg_strategy(),
        prover_verifier_config_instance
            .1
            .get_claim_agg_constant_column_optimization(),
    )
}

// -------------------- Helper functions for checking prover/verifier readiness --------------------
impl GKRCircuitProverConfig {
    /// Returns whether the calling [GKRCircuitProverConfig], i.e. the "expected"
    /// config, matches the one which is currently configured globally. A match
    /// implies that we are ready to start proving.
    ///
    /// Note that the `prover_verifier_config_instance` should be held for
    /// the ENTIRE DURATION of proving (and verifying) to prevent any changes
    /// from happening to the config in the middle of proving!
    pub fn matches_global_prover_config(
        &self,
        prover_verifier_config_instance: &RwLockReadGuard<
            '_,
            (GKRCircuitProverConfig, GKRCircuitVerifierConfig),
        >,
    ) -> bool {
        self.get_claim_agg_constant_column_optimization()
            == prover_verifier_config_instance
                .0
                .get_claim_agg_constant_column_optimization()
            && self.get_enable_bit_packing()
                == prover_verifier_config_instance.0.get_enable_bit_packing()
            && self.get_lazy_beta_evals() == prover_verifier_config_instance.0.get_lazy_beta_evals()
            && self.get_circuit_description_hash_type()
                == prover_verifier_config_instance
                    .0
                    .get_circuit_description_hash_type()
            && self.get_claim_agg_strategy()
                == prover_verifier_config_instance.0.get_claim_agg_strategy()
    }
}

impl GKRCircuitVerifierConfig {
    /// Returns whether the calling [GKRCircuitVerifierConfig], i.e. the "expected"
    /// config, matches the one which is currently configured globally. A match
    /// implies that we are ready to start verifying.
    ///
    /// Note that the `prover_verifier_config_instance` should be held for
    /// the ENTIRE DURATION of verifying to prevent any changes
    /// from happening to the config in the middle of verifying!
    pub fn matches_global_verifier_config(
        &self,
        prover_verifier_config_instance: &RwLockReadGuard<
            '_,
            (GKRCircuitProverConfig, GKRCircuitVerifierConfig),
        >,
    ) -> bool {
        self.get_claim_agg_constant_column_optimization()
            == prover_verifier_config_instance
                .1
                .get_claim_agg_constant_column_optimization()
            && self.get_lazy_beta_evals() == prover_verifier_config_instance.1.get_lazy_beta_evals()
            && self.get_circuit_description_hash_type()
                == prover_verifier_config_instance
                    .1
                    .get_circuit_description_hash_type()
            && self.get_claim_agg_strategy()
                == prover_verifier_config_instance.1.get_claim_agg_strategy()
    }

    /// Returns whether the current (expected) [GKRCircuitVerifierConfig] matches
    /// the [ProofConfig] for the given proof.
    pub fn matches_proof_config(&self, proof_config: &ProofConfig) -> bool {
        self.get_claim_agg_constant_column_optimization()
            == proof_config.get_claim_agg_constant_column_optimization()
            && self.get_circuit_description_hash_type()
                == proof_config.get_circuit_description_hash_type()
            && self.get_claim_agg_strategy() == proof_config.get_claim_agg_strategy()
    }
}

// -------------------- Wrapper fns for ease of non-import --------------------
/// Simple wrapper around [parking_lot::RwLockWriteGuard::downgrade] so that
/// dependent libraries don't have to include the dependency within their
/// `Cargo.toml` files.
pub fn downgrade<T>(write_guard: RwLockWriteGuard<'_, T>) -> RwLockReadGuard<'_, T> {
    RwLockWriteGuard::downgrade(write_guard)
}

// -------------------- Helper fns for execution --------------------

/// Similar function to [perform_function_under_expected_configs], but only
/// checks against an expected [GKRCircuitProverConfig].
#[macro_export]
macro_rules! perform_function_under_prover_config {
    ($func:expr, $expected_prover_config:expr, $($arg:expr),*) => {{

        loop {
            // Additional scope so that the read lock is dropped immediately
            {
                // Attempt to get read lock and check prover/verifier config matches
                let prover_verifier_static_config_read_instance = remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.read();
                if $expected_prover_config
                    .matches_global_prover_config(&prover_verifier_static_config_read_instance)
                {
                    // Execute the function with the provided arguments
                    let ret = $func($($arg),*);
                    break ret;
                }
            }

            // Otherwise, write the desired prover/verifier config to the global static var
            if let Some(mut prover_verifier_static_config_write_instance) =
                remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.try_write()
            {
                remainder_shared_types::config::global_config::set_global_prover_config(
                    $expected_prover_config,
                    &mut prover_verifier_static_config_write_instance,
                );

                // Downgrade into a read lock and perform the function (this should unblock other readers)
                let _prover_verifier_static_config_read_instance =
                    remainder_shared_types::config::global_config::downgrade(prover_verifier_static_config_write_instance);

                // Execute the function with the provided arguments
                let ret = $func($($arg),*);
                break ret;
            }
        }
    }};
}

/// Similar function to [perform_function_under_expected_configs], but only
/// checks against an expected [GKRCircuitVerifierConfig].
#[macro_export]
macro_rules! perform_function_under_verifier_config {
    ($func:expr, $expected_verifier_config:expr, $($arg:expr),*) => {{

        loop {
            // Additional scope so that the read lock is dropped immediately
            {
                // Attempt to get read lock and check prover/verifier config matches
                let prover_verifier_static_config_read_instance = remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.read();
                if $expected_verifier_config
                    .matches_global_verifier_config(&prover_verifier_static_config_read_instance)
                {
                    let ret = $func($($arg),*);
                    break ret;
                }
            }

            // Otherwise, write the desired prover/verifier config to the global static var
            if let Some(mut prover_verifier_static_config_write_instance) =
                remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.try_write()
            {
                remainder_shared_types::config::global_config::set_global_verifier_config(
                    $expected_verifier_config,
                    &mut prover_verifier_static_config_write_instance,
                );

                // Downgrade into a read lock and perform the function (this should unblock other readers)
                let _prover_verifier_static_config_read_instance =
                    remainder_shared_types::config::global_config::downgrade(prover_verifier_static_config_write_instance);

                // Execute the function with the provided arguments
                let ret = $func($($arg),*);
                break ret;
            }
        }
    }};
}

/// This function will run the given function _only_ under the given configs!
/// It does this by first reading the global config and checking whether
/// that config matches the expected configs which are passed in, then
/// * If matches, it performs the function immediately while holding the read
///     lock and returns.
/// * If doesn't match, it attempts to acquire a write lock, then writes the
///     config to the global one, downgrades the write lock to a read lock
///     (thereby unblocking all other readers), and performs the function.
#[macro_export]
macro_rules! perform_function_under_expected_configs {
    ($func:expr, $expected_prover_config: expr, $expected_verifier_config:expr, $($arg:expr),*) => {{

        loop {
            // Extra scope allows us to automatically drop the reader reference
            {
                // Attempt to get read lock and check prover/verifier config matches
                let prover_verifier_static_config_read_instance = remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.read();
                if $expected_prover_config
                    .matches_global_prover_config(&prover_verifier_static_config_read_instance)
                    && $expected_verifier_config
                        .matches_global_verifier_config(&prover_verifier_static_config_read_instance)
                {
                    let ret = $func($($arg),*);
                    break ret;
                }
            }

            // Otherwise, write the desired prover/verifier config to the global static var, downgrade into a read lock, and perform the action
            if let Some(mut prover_verifier_static_config_write_instance) =
                remainder_shared_types::config::global_config::PROVER_VERIFIER_CONFIG.try_write()
            {
                remainder_shared_types::config::global_config::set_global_prover_config(
                    $expected_prover_config,
                    &mut prover_verifier_static_config_write_instance,
                );
                remainder_shared_types::config::global_config::set_global_verifier_config(
                    $expected_verifier_config,
                    &mut prover_verifier_static_config_write_instance,
                );

                // Downgrade into a read lock and perform the function (this should unblock other readers)
                let _prover_verifier_static_config_read_instance =
                    remainder_shared_types::config::global_config::downgrade(prover_verifier_static_config_write_instance);

                // Execute the function with the provided arguments
                let ret = $func($($arg),*);
                break ret;
            }
        }
    }};
}

// pub fn perform_function_under_expected_configs<F, A, R>(
//     f: F,
//     args: A,
//     expected_prover_config: &GKRCircuitProverConfig,
//     expected_verifier_config: &GKRCircuitVerifierConfig,
// ) -> R
// where
//     F: FnOnce(A) -> R,
// {
//     loop {
//         // --- Extra scope allows us to automatically drop the reader reference ---
//         {
//             // --- Attempt to get read lock and check prover/verifier config matches ---
//             let prover_verifier_static_config_read_instance = PROVER_VERIFIER_CONFIG.read();
//             if expected_prover_config
//                 .matches_global_prover_config(&prover_verifier_static_config_read_instance)
//                 && expected_verifier_config
//                     .matches_global_verifier_config(&prover_verifier_static_config_read_instance)
//             {
//                 let ret = f(args);
//                 return ret;
//             }
//         }

//         // --- Otherwise, write the desired prover/verifier config to the global static var, downgrade into a read lock, and perform the action ---
//         if let Some(mut prover_verifier_static_config_write_instance) =
//             PROVER_VERIFIER_CONFIG.try_write()
//         {
//             set_global_prover_config(
//                 expected_prover_config,
//                 &mut prover_verifier_static_config_write_instance,
//             );
//             set_global_verifier_config(
//                 expected_verifier_config,
//                 &mut prover_verifier_static_config_write_instance,
//             );

//             // Downgrade into a read lock and perform the function (this should unblock other readers)
//             let _prover_verifier_static_config_read_instance =
//                 RwLockWriteGuard::downgrade(prover_verifier_static_config_write_instance);

//             let ret = f(args);
//             return ret;
//         }
//     }
// }

// /// Similar function to [perform_function_under_expected_configs], but only
// /// checks against an expected [GKRCircuitProverConfig].
// pub fn perform_function_under_prover_config<F, A, R>(
//     f: F,
//     args: A,
//     expected_prover_config: &GKRCircuitProverConfig,
// ) -> R
// where
//     F: FnOnce(A) -> R,
// {
//     loop {
//         // --- Additional scope so that the read lock is dropped immediately ---
//         {
//             // --- Attempt to get read lock and check prover/verifier config matches ---
//             let prover_verifier_static_config_read_instance = PROVER_VERIFIER_CONFIG.read();
//             if expected_prover_config
//                 .matches_global_prover_config(&prover_verifier_static_config_read_instance)
//             {
//                 let ret = f(args);
//                 return ret;
//             }
//         }

//         // --- Otherwise, write the desired prover/verifier config to the global static var ---
//         if let Some(mut prover_verifier_static_config_write_instance) =
//             PROVER_VERIFIER_CONFIG.try_write()
//         {
//             set_global_prover_config(
//                 expected_prover_config,
//                 &mut prover_verifier_static_config_write_instance,
//             );

//             // Downgrade into a read lock and perform the function (this should unblock other readers)
//             let _prover_verifier_static_config_read_instance =
//                 RwLockWriteGuard::downgrade(prover_verifier_static_config_write_instance);

//             let ret = f(args);
//             return ret;
//         }
//     }
// }

// /// Similar function to [perform_function_under_expected_configs], but only
// /// checks against an expected [GKRCircuitVerifierConfig].
// pub fn perform_function_under_verifier_config<F, A, R>(
//     f: F,
//     args: A,
//     expected_verifier_config: &GKRCircuitVerifierConfig,
// ) -> R
// where
//     F: FnOnce(A) -> R,
// {
//     loop {
//         // --- Additional scope so that the read lock is dropped immediately ---
//         {
//             // --- Attempt to get read lock and check prover/verifier config matches ---
//             let prover_verifier_static_config_read_instance = PROVER_VERIFIER_CONFIG.read();
//             if expected_verifier_config
//                 .matches_global_verifier_config(&prover_verifier_static_config_read_instance)
//             {
//                 let ret = f(args);
//                 return ret;
//             }
//         }

//         // --- Otherwise, write the desired prover/verifier config to the global static var ---
//         if let Some(mut prover_verifier_static_config_write_instance) =
//             PROVER_VERIFIER_CONFIG.try_write()
//         {
//             set_global_verifier_config(
//                 expected_verifier_config,
//                 &mut prover_verifier_static_config_write_instance,
//             );

//             // Downgrade into a read lock and perform the function (this should unblock other readers)
//             let _prover_verifier_static_config_read_instance =
//                 RwLockWriteGuard::downgrade(prover_verifier_static_config_write_instance);

//             let ret = f(args);
//             return ret;
//         }
//     }
// }

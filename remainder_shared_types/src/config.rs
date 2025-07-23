use crate::circuit_hash::CircuitHashType;
use serde::{Deserialize, Serialize};
pub mod global_config;

// ------------------ Circuit-specific (GKR) proving ------------------
/// An enum listing the types of claim aggregation strategies.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ClaimAggregationStrategy {
    /// Interpolation-based claim aggregation strategy from "Thaler13".
    Interpolative,
    /// Claim aggregation using the random-linear combination strategy from "Libra"
    RLC,
}

/// A config which informs a GKR circuit prover about how to prove a GKR circuit,
/// including flags e.g. whether to use certain optimizations (e.g. `Evaluations`
/// memory-efficient optimization), or e.g. which claim aggregation strategy to use.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GKRCircuitProverConfig {
    /// Whether to evaluate the beta function within gate layers lazily,
    /// i.e. to compute the values on the fly rather than via initializing
    /// a beta evaluations "table".
    lazy_beta_evals: bool,

    /// The type of hash function to be used on hashing the circuit description
    /// to be added to transcript.
    circuit_description_hash_type: CircuitHashType,

    /// Which claim aggregation (RLC vs. deterministic) to use for reducing
    /// the validity of multiple claims on a layer to that of a single claim.
    claim_agg_strategy: ClaimAggregationStrategy,

    /// Whether to use the "constant column optimization", i.e. whether to
    /// reduce the implicit degree of the prover-claimed polynomial
    /// Q(x) =? V_i(l(x)) when there are variable indices within all claims
    /// where all challenges within that index are identical.
    claim_agg_constant_column_optimization: bool,

    /// Hyrax input layer batch opening. Determines whether we attempt to aggregate
    /// Hyrax PCS Evaluation proofs by grouping common challenge coordinates together.
    hyrax_input_layer_batch_opening: bool,

    /// Controls whether bit-packing is actually enabled. If set to `false`, the
    /// `BitPackedVector` will default to storing each field element using the type
    /// `F`, effectively behaving like a regular (immutable) `Vec<F>`. This is
    /// needed because bit-packing incurs a noticable runtime slowdown, and we need
    /// an easy way to turn it off if trading memory for speed is desirable.
    ///
    /// Note that because this is global, this option also implicitly affects
    /// the verifier's `Evaluations<F>` structs!!!
    enable_bit_packing: bool,
}

impl GKRCircuitProverConfig {
    /// Returns a memory-optimal configuration for the GKR circuit prover.
    ///
    /// In particular, this turns on the `lazy_beta_evals` and `bit_packed_vector`
    /// flags.
    pub fn memory_optimized_default() -> Self {
        Self {
            lazy_beta_evals: true,
            circuit_description_hash_type: CircuitHashType::Sha3_256,
            claim_agg_strategy: ClaimAggregationStrategy::Interpolative,
            claim_agg_constant_column_optimization: true,
            hyrax_input_layer_batch_opening: true,
            enable_bit_packing: true,
        }
    }

    /// Returns a runtime-optimal configuration for the GKR circuit prover.
    pub fn runtime_optimized_default() -> Self {
        Self {
            lazy_beta_evals: false,
            circuit_description_hash_type: CircuitHashType::Sha3_256,
            claim_agg_strategy: ClaimAggregationStrategy::RLC,
            claim_agg_constant_column_optimization: true,
            hyrax_input_layer_batch_opening: true,
            enable_bit_packing: false,
        }
    }

    /// Returns a runtime-optimal configuration for a version of the GKR
    /// circuit prover which is compatible with our current Hyrax IP
    /// implementation.
    ///
    /// In particular, this turns OFF the `enable_bit_packing` option, turns OFF
    /// the `lazy_beta_evals` option, and additionally turns OFF the
    /// `claim_agg_constant_column_optimization` option as well.
    pub const fn hyrax_compatible_runtime_optimized_default() -> Self {
        Self {
            lazy_beta_evals: false,
            circuit_description_hash_type: CircuitHashType::Sha3_256,
            claim_agg_strategy: ClaimAggregationStrategy::RLC,
            claim_agg_constant_column_optimization: false,
            hyrax_input_layer_batch_opening: true,
            enable_bit_packing: false,
        }
    }

    /// Returns a memory-optimal configuration for a version of the GKR
    /// circuit prover which is compatible with our current Hyrax IP
    /// implementation.
    ///
    /// In particular, this turns ON the `enable_bit_packing` option, turns ON
    /// the `lazy_beta_evals`, and additionally turns OFF the
    /// `claim_agg_constant_column_optimization` option as well.
    pub const fn hyrax_compatible_memory_optimized_default() -> Self {
        Self {
            lazy_beta_evals: true,
            circuit_description_hash_type: CircuitHashType::Sha3_256,
            claim_agg_strategy: ClaimAggregationStrategy::RLC,
            claim_agg_constant_column_optimization: false,
            hyrax_input_layer_batch_opening: true,
            enable_bit_packing: true,
        }
    }

    /// Constructs a new [GKRCircuitProverConfig] from scratch.
    pub fn new(
        lazy_beta_evals: bool,
        circuit_description_hash_type: CircuitHashType,
        claim_agg_strategy: ClaimAggregationStrategy,
        claim_agg_constant_column_optimization: bool,
        hyrax_input_layer_batch_opening: bool,
        enable_bit_packing: bool,
    ) -> Self {
        Self {
            lazy_beta_evals,
            circuit_description_hash_type,
            claim_agg_strategy,
            claim_agg_constant_column_optimization,
            hyrax_input_layer_batch_opening,
            enable_bit_packing,
        }
    }

    /// Setter function for lazy beta evals.
    pub fn set_lazy_beta_evals(&mut self, updated_lazy_beta_evals: bool) {
        self.lazy_beta_evals = updated_lazy_beta_evals;
    }

    /// Setter function for claim agg strategy.
    pub fn set_claim_agg_strategy(&mut self, updated_claim_agg_strategy: ClaimAggregationStrategy) {
        self.claim_agg_strategy = updated_claim_agg_strategy;
    }

    /// Setter function for circuit hash type.
    pub fn set_circuit_description_hash_type(
        &mut self,
        updated_circuit_description_hash_type: CircuitHashType,
    ) {
        self.circuit_description_hash_type = updated_circuit_description_hash_type;
    }

    /// Setter function for constant column optimization.
    pub fn set_claim_agg_constant_column_optimization(
        &mut self,
        updated_claim_agg_constant_column_optimization: bool,
    ) {
        self.claim_agg_constant_column_optimization =
            updated_claim_agg_constant_column_optimization;
    }

    /// Setter function for enabling bit packing.
    pub fn set_enable_bit_packing(&mut self, updated_enable_bit_packing: bool) {
        self.enable_bit_packing = updated_enable_bit_packing;
    }

    /// Getter function for lazy beta evals.
    pub fn get_lazy_beta_evals(&self) -> bool {
        self.lazy_beta_evals
    }

    /// Getter function for claim agg strategy.
    pub fn get_claim_agg_strategy(&self) -> ClaimAggregationStrategy {
        self.claim_agg_strategy
    }

    /// Getter function for circuit hash type.
    pub fn get_circuit_description_hash_type(&self) -> CircuitHashType {
        self.circuit_description_hash_type
    }

    /// Getter function for constant column optimization.
    pub fn get_claim_agg_constant_column_optimization(&self) -> bool {
        self.claim_agg_constant_column_optimization
    }

    /// Getter function for Hyrax batch opening.
    pub fn get_hyrax_batch_opening(&self) -> bool {
        self.hyrax_input_layer_batch_opening
    }

    /// Getter function for enabling bit packing.
    pub fn get_enable_bit_packing(&self) -> bool {
        self.enable_bit_packing
    }
}

/// A config which informs a GKR circuit verifier about how to verify a GKR circuit + proof,
/// including flags e.g. how to aggregate claims.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GKRCircuitVerifierConfig {
    /// Whether to evaluate the beta function within gate layers lazily,
    /// i.e. to compute the values on the fly rather than via initializing
    /// a beta evaluations "table".
    lazy_beta_evals: bool,

    /// The type of hash function to be used for hashing the circuit description
    /// to be added to transcript.
    circuit_description_hash_type: CircuitHashType,

    /// Which claim aggregation (RLC vs. deterministic) to use for reducing
    /// the validity of multiple claims on a layer to that of a single claim.
    claim_agg_strategy: ClaimAggregationStrategy,

    /// Whether to use the "constant column optimization", i.e. whether to
    /// reduce the implicit degree of the prover-claimed polynomial
    /// Q(x) =? V_i(l(x)) when there are variable indices within all claims
    /// where all challenges within that index are identical.
    claim_agg_constant_column_optimization: bool,
}

impl GKRCircuitVerifierConfig {
    /// Constructs a [GKRCircuitVerifierConfig] from the [GKRCircuitProverConfig]
    /// used for the corresponding prover.
    pub fn new_from_prover_config(
        prover_config: &GKRCircuitProverConfig,
        verifier_lazy_beta_evals: bool,
    ) -> Self {
        Self {
            lazy_beta_evals: verifier_lazy_beta_evals,
            claim_agg_strategy: prover_config.claim_agg_strategy,
            circuit_description_hash_type: prover_config.circuit_description_hash_type,
            claim_agg_constant_column_optimization: prover_config
                .claim_agg_constant_column_optimization,
        }
    }

    /// Constructs a new [GKRCircuitVerifierConfig] from scratch.
    pub const fn new(
        lazy_beta_evals: bool,
        circuit_description_hash_type: CircuitHashType,
        claim_agg_strategy: ClaimAggregationStrategy,
        claim_agg_constant_column_optimization: bool,
    ) -> Self {
        Self {
            lazy_beta_evals,
            circuit_description_hash_type,
            claim_agg_strategy,
            claim_agg_constant_column_optimization,
        }
    }

    /// Constructs a [GKRCircuitVerifierConfig] from the corresponding
    /// [ProofConfig] used within the proof.
    pub fn new_from_proof_config(
        proof_config: &ProofConfig,
        verifier_lazy_beta_evals: bool,
    ) -> Self {
        Self {
            lazy_beta_evals: verifier_lazy_beta_evals,
            claim_agg_strategy: proof_config.claim_agg_strategy,
            claim_agg_constant_column_optimization: proof_config
                .claim_agg_constant_column_optimization,
            circuit_description_hash_type: proof_config.circuit_description_hash_type,
        }
    }

    /// Returns a runtime-optimal configuration for a version of the GKR
    /// circuit verifier compatible with Hyrax.
    ///
    /// In particular, this turns OFF the `lazy_beta_evals`, and additionally
    /// turns OFF the `claim_agg_constant_column_optimization` option as well.
    pub const fn hyrax_compatible_runtime_optimized_default() -> Self {
        Self {
            lazy_beta_evals: false,
            circuit_description_hash_type: CircuitHashType::Sha3_256,
            claim_agg_strategy: ClaimAggregationStrategy::Interpolative,
            claim_agg_constant_column_optimization: false,
        }
    }

    /// Setter function for lazy beta evals.
    pub fn set_lazy_beta_evals(&mut self, updated_lazy_beta_evals: bool) {
        self.lazy_beta_evals = updated_lazy_beta_evals;
    }

    /// Setter function for claim agg strategy.
    pub fn set_claim_agg_strategy(&mut self, updated_claim_agg_strategy: ClaimAggregationStrategy) {
        self.claim_agg_strategy = updated_claim_agg_strategy;
    }

    /// Setter function for circuit hash type.
    pub fn set_circuit_description_hash_type(
        &mut self,
        updated_circuit_description_hash_type: CircuitHashType,
    ) {
        self.circuit_description_hash_type = updated_circuit_description_hash_type;
    }

    /// Setter function for constant column optimization.
    pub fn set_claim_agg_constant_column_optimization(
        &mut self,
        updated_claim_agg_constant_column_optimization: bool,
    ) {
        self.claim_agg_constant_column_optimization =
            updated_claim_agg_constant_column_optimization;
    }

    /// Getter function for lazy beta evals.
    pub fn get_lazy_beta_evals(&self) -> bool {
        self.lazy_beta_evals
    }

    /// Getter function for claim agg strategy.
    pub fn get_claim_agg_strategy(&self) -> ClaimAggregationStrategy {
        self.claim_agg_strategy
    }

    /// Getter function for circuit hash type.
    pub fn get_circuit_description_hash_type(&self) -> CircuitHashType {
        self.circuit_description_hash_type
    }

    /// Getter function for constant column optimization.
    pub fn get_claim_agg_constant_column_optimization(&self) -> bool {
        self.claim_agg_constant_column_optimization
    }
}

// -------------------- Proof config --------------------

/// A config which travels alongside a GKR proof stored within a `Transcript`
/// which details the proof-specific configuration (i.e. how the verifier
/// should be configured in order to appropriately parse and verify the proof).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProofConfig {
    /// The type of hash function to be used on hashing the circuit description
    /// to be added to transcript.
    circuit_description_hash_type: CircuitHashType,

    /// Which claim aggregation (RLC vs. deterministic) to use for reducing
    /// the validity of multiple claims on a layer to that of a single claim.
    claim_agg_strategy: ClaimAggregationStrategy,

    /// Whether to use the "constant column optimization", i.e. whether to
    /// reduce the implicit degree of the prover-claimed polynomial
    /// Q(x) =? V_i(l(x)) when there are variable indices within all claims
    /// where all challenges within that index are identical.
    claim_agg_constant_column_optimization: bool,
}
impl ProofConfig {
    /// Creates the associated [ProofConfig] from a [GKRCircuitProverConfig].
    /// Note that a similar function should not be required for a
    /// [GKRCircuitVerifierConfig], since only the circuit prover should specify
    /// the proof config and the verifier should simply check to see if their
    /// own config matches that of the proof given.
    pub fn new_from_prover_config(prover_config: &GKRCircuitProverConfig) -> Self {
        Self {
            circuit_description_hash_type: prover_config.circuit_description_hash_type,
            claim_agg_strategy: prover_config.claim_agg_strategy,
            claim_agg_constant_column_optimization: prover_config
                .claim_agg_constant_column_optimization,
        }
    }

    /// Setter function for claim agg strategy.
    pub fn set_claim_agg_strategy(&mut self, updated_claim_agg_strategy: ClaimAggregationStrategy) {
        self.claim_agg_strategy = updated_claim_agg_strategy;
    }

    /// Setter function for circuit hash type.
    pub fn set_circuit_description_hash_type(
        &mut self,
        updated_circuit_description_hash_type: CircuitHashType,
    ) {
        self.circuit_description_hash_type = updated_circuit_description_hash_type;
    }

    /// Setter function for constant column optimization.
    pub fn set_claim_agg_constant_column_optimization(
        &mut self,
        updated_claim_agg_constant_column_optimization: bool,
    ) {
        self.claim_agg_constant_column_optimization =
            updated_claim_agg_constant_column_optimization;
    }

    /// Getter function for claim agg strategy.
    pub fn get_claim_agg_strategy(&self) -> ClaimAggregationStrategy {
        self.claim_agg_strategy
    }

    /// Getter function for circuit hash type.
    pub fn get_circuit_description_hash_type(&self) -> CircuitHashType {
        self.circuit_description_hash_type
    }

    /// Getter function for constant column optimization.
    pub fn get_claim_agg_constant_column_optimization(&self) -> bool {
        self.claim_agg_constant_column_optimization
    }
}

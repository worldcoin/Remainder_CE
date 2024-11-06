//! For assigning ids to nodes and various sorts of layers.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// [CircuitBuildingContext] keeps track of the ids of the nodes and layers that are created.
/// Contains a consistently incrementing counters to prevent id collisions.
#[derive(Debug, Default)]
pub struct CircuitBuildingContext {
    node_id: AtomicU64,
    input_layer_id: AtomicUsize,
    layer_id: AtomicUsize,
    fiat_shamir_challenge_layer_id: AtomicUsize,
}

static CONTEXT: Lazy<CircuitBuildingContext> = Lazy::new(|| CircuitBuildingContext {
    node_id: AtomicU64::new(0),
    input_layer_id: AtomicUsize::new(0),
    layer_id: AtomicUsize::new(0),
    fiat_shamir_challenge_layer_id: AtomicUsize::new(0),
});

impl CircuitBuildingContext {
    /// Retrieves a new node id that is guaranteed to be unique.
    pub fn next_node_id() -> u64 {
        CONTEXT.node_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new input layer id that is guaranteed to be unique.
    pub fn next_input_layer_id() -> usize {
        CONTEXT.input_layer_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new layer id that is guaranteed to be unique.
    pub fn next_layer_id() -> usize {
        CONTEXT.layer_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new fiat shamir challenge layer id that is guaranteed to be unique.
    pub fn next_fiat_shamir_challenge_layer_id() -> usize {
        CONTEXT
            .fiat_shamir_challenge_layer_id
            .fetch_add(1, Ordering::Relaxed)
    }
}

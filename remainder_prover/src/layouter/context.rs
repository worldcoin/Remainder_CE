//! For assigning ids to nodes and various sorts of layers.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use once_cell::sync::Lazy;

/// Container of global context for node and layer creation.
/// Contains a consistently incrementing counters to prevent id collisions.
#[derive(Debug, Default)]
pub struct CircuitContext {
    node_id: AtomicUsize,
    input_layer_id: AtomicU64,
    layer_id: AtomicU64,
    output_layer_id: AtomicU64,
}

static CONTEXT: Lazy<CircuitContext> = Lazy::new(|| CircuitContext {
    node_id: AtomicUsize::new(0),
    input_layer_id: AtomicU64::new(0),
    layer_id: AtomicU64::new(0),
    output_layer_id: AtomicU64::new(0),
});

impl CircuitContext {
    /// Resets the context to its initial state, such that the next ids to be issued are all 0.
    pub fn reset() {
        CONTEXT.node_id.store(0, Ordering::SeqCst);
        CONTEXT.input_layer_id.store(0, Ordering::SeqCst);
        CONTEXT.layer_id.store(0, Ordering::SeqCst);
        CONTEXT.output_layer_id.store(0, Ordering::SeqCst);
    }

    /// Retrieves a new node id that is guaranteed to be unique.
    pub fn next_node_id() -> usize {
        CONTEXT.node_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new input layer id that is guaranteed to be unique.
    pub fn next_input_layer_id() -> u64 {
        CONTEXT.input_layer_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new layer id that is guaranteed to be unique.
    pub fn next_layer_id() -> u64 {
        CONTEXT.layer_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Retrieves a new output layer id that is guaranteed to be unique.
    pub fn next_output_layer_id() -> u64 {
        CONTEXT.output_layer_id.fetch_add(1, Ordering::Relaxed)
    }
}
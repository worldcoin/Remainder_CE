use crate::FieldExt;
use derive_more::{From, Into};

/// --- Path nodes within the tree and in the path hint ---
/// Used for the following components of the (circuit) input:
/// a) 
#[derive(Copy, Debug, Clone, From, Into)]
pub struct DecisionNode<F: FieldExt> {
    pub node_id: F,
    pub attr_id: F,
    pub threshold: F,
}

#[derive(Copy, Debug, Clone, From, Into)]
pub struct LeafNode<F: FieldExt> {
    pub node_id: F,
    pub node_val: F,
}

/// --- 16-bit binary decomposition ---
/// Used for the following components of the (circuit) input:
/// a) The binary decomposition of the path node hints (i.e. path_x.thr - x.val)
/// b) The binary decomposition of the multiplicity coefficients $c_j$
#[derive(Copy, Debug, Clone, From, Into)]
pub struct BinDecomp16Bit<F: FieldExt> {
    pub bits: [F; 16],
}

/// --- Input element to the tree, i.e. a list of input attributes ---
/// Used for the following components of the (circuit) input:
/// a) The actual input attributes, i.e. x
/// b) The permuted input attributes, i.e. \bar{x}
#[derive(Copy, Debug, Clone, From, Into, PartialEq)]
pub struct InputAttribute<F: FieldExt> {
    // pub attr_idx: F,
    pub attr_id: F,
    pub attr_val: F,
}

// --- Just an enumeration of, uh, stuff...? ---
// To be honest this is basically just DenseMle<F>
// TODO!(ryancao)
// #[derive(Debug, Clone)]
// pub struct EnumerationRange<F: FieldExt> {
//     // TODO!(ryancao)
//     pub attr_id: F,
//     // pub attr_val: F,
// }

// Personally for the above, just give me a Vec<u32> and that should be great!
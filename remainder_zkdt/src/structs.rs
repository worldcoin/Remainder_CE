use remainder::mle::circuit_mle::FlatMles;
use remainder_shared_types::Field;
use serde::{Deserialize, Serialize};

// ------------------------------------ ACTUAL DATA STRUCTS ------------------------------------

/// --- Input element to the tree, i.e. a list of input attributes ---
/// Used for the following components of the (circuit) input:
/// a) The actual input attributes, i.e. x
/// b) The permuted input attributes, i.e. \bar{x}
#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAttribute<F> {
    // pub attr_idx: F,
    ///The attr id of this input
    pub attr_id: F,
    ///The threshold value of this input
    pub attr_val: F,
}
pub type InputAttributeMle<F> = FlatMles<F, 2>;

/// --- Path nodes within the tree and in the path hint ---
#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct DecisionNode<F> {
    ///The id of this node in the tree
    pub(crate) node_id: F,
    ///The id of the attribute this node involves
    pub(crate) attr_id: F,
    ///The treshold of this node
    pub(crate) threshold: F,
}
pub type DecisionNodeMle<F> = FlatMles<F, 3>;

#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
///The Leafs of the tree
pub struct LeafNode<F> {
    ///The id of this leaf in the tree
    pub(crate) node_id: F,
    ///The value of this leaf
    pub(crate) node_val: F,
}
pub type LeafNodeMle<F> = FlatMles<F, 2>;

#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Used for the attribute multiplicities
pub struct BinDecomp8Bit<F> {
    ///The 8 bits that make up this decomposition
    ///
    /// Should all be 1 or 0
    pub bits: [F; 8],
}
pub type BinDecomp8BitMle<F> = FlatMles<F, 8>;

/// --- 16-bit binary decomposition ---
/// Used for the following components of the (circuit) input:
/// a) The binary decomposition of the path node hints (i.e. x.val - path_x.thr)
/// b) The binary decomposition of the multiplicity coefficients $c_j$
#[derive(Copy, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinDecomp16Bit<F> {
    ///The 16 bits that make up this decomposition
    ///
    /// Should all be 1 or 0
    pub bits: [F; 16],
}
pub type BinDecomp16BitMle<F> = FlatMles<F, 16>;

// ------------------------------------ FROM IMPL FOR BINDECOMP ------------------------------------

impl<F: Field> From<Vec<bool>> for BinDecomp16Bit<F> {
    fn from(bits: Vec<bool>) -> Self {
        BinDecomp16Bit::<F> {
            bits: bits
                .iter()
                .map(|x| F::from(*x as u64))
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<F: Field> From<Vec<bool>> for BinDecomp8Bit<F> {
    fn from(bits: Vec<bool>) -> Self {
        BinDecomp8Bit::<F> {
            bits: bits
                .iter()
                .map(|x| F::from(*x as u64))
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        }
    }
}

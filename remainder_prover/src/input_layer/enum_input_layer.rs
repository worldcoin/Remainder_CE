//! A wrapper type that makes working with variants of InputLayer easier.

use remainder_shared_types::Field;

use crate::{
    claims::wlx_eval::YieldWLXEvals,
    mle::dense::DenseMle,
};

use super::{
    ligero_input_layer::LigeroInputLayer,
    public_input_layer::PublicInputLayer
};
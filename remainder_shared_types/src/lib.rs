pub mod transcript;

use halo2_base::utils::ScalarField;
use serde::{Serialize, Deserialize};

///External definition of Field element trait, will remain an Alias for now
pub trait FieldExt: ScalarField + Serialize + for <'de> Deserialize<'de> {}

impl<F: ScalarField + Serialize + for <'de> Deserialize<'de>> FieldExt for F {}
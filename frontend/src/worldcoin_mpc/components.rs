use shared_types::Field;

use crate::{
    layouter::builder::{CircuitBuilder, NodeRef},
    worldcoin_mpc::parameters::GR4_MODULUS,
};

/// Components used by worldcoin mpc
pub struct WorldcoinMpcComponents;

impl WorldcoinMpcComponents {
    /// Calculates masked iris code from iris code and mask, making the result
    /// available as self.sector.
    ///
    /// Spec: --  iris_code in {0, 1}
    ///       --  mask in {0, 1}
    ///       --  masked_iris_code in {-1, 0, 1}, where 0 means that the bit was
    ///           masked and {-1, 1} correspond to {0, 1} iris_code responses
    ///
    /// It is assumed that `iris_code` and `mask` have the same length.
    ///
    /// the expression is `mask - 2 * (iris_code * mask)`, a.k.a
    /// (-2) * iris_code - mask
    ///
    /// see notion page: Worldcoin specification: iris code versions and Hamming distance
    pub fn masked_iris_code<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        iris_code: &NodeRef<F>,
        mask: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(-(iris_code * F::from(2)) - mask + F::from(GR4_MODULUS))
    }

    /// Calculates `iris_code + evaluation_point_times_slopes`, making the result available as self.sector.
    /// It is assumed that `iris_code` and `evaluation_point_times_slopes` have the same length.
    pub fn sum<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        encoded_masked_iris_code: &NodeRef<F>,
        evaluation_point_times_slopes: &NodeRef<F>,
    ) -> NodeRef<F> {
        builder_ref.add_sector(encoded_masked_iris_code + evaluation_point_times_slopes)
    }

    /// Checks `computed_shares` and `shared_moduled` are the same modulo const `GR4_MODULO`.
    /// Uses an auxilary `quotient` to store their difference divided by `GR4_MODULO`.
    /// Makes the result available as self.sector. Also makes a zero output layer.
    /// It is assumed that `computed_shares` and `shared_moduled` have the same length.
    pub fn congruence<F: Field>(
        builder_ref: &mut CircuitBuilder<F>,
        quotient: &NodeRef<F>,
        computed_shares: &NodeRef<F>,
        shares_reduced_modulo_gr4_modulus: &NodeRef<F>,
    ) -> NodeRef<F> {
        let sector = builder_ref.add_sector(
            quotient * F::from(GR4_MODULUS) + shares_reduced_modulo_gr4_modulus - computed_shares,
        );

        builder_ref.set_output(&sector);
        sector
    }
}

use std::marker::PhantomData;

use remainder_shared_types::FieldExt;

use crate::{
    builders::layer_builder::LayerBuilder,
    layer::{
        gate::{BinaryOperation, Gate},
        regular_layer::RegularLayer,
        Layer, LayerId,
    },
    mle::{
        dense::{DenseMle, DenseMleRef},
        MleRef,
    },
};

/// The list of Layers that make up the GKR circuit
pub struct Layers<F: FieldExt, T: Layer<F>> {
    /// A Vec of pointers to various layer types
    pub layers: Vec<T>,
    marker: PhantomData<F>,
}

impl<F: FieldExt, T: Layer<F>> Layers<F, T> {
    /// Add a GKRLayer to a list of layers
    pub fn add_gkr<B: LayerBuilder<F>>(&mut self, new_layer: B) -> B::Successor
    where
        T: From<RegularLayer<F>>,
    {
        let id = LayerId::Layer(self.layers.len());
        let successor = new_layer.next_layer(id, None);
        let layer = RegularLayer::<F>::new(new_layer, id);
        self.layers.push(layer.into());
        successor
    }

    /// Add a batched Add Gate layer to a list of layers
    /// In the batched case, consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Add a Gate layer to a list of layers
    /// In the batched case (`num_dataparallel_bits` > 0), consider a vector of mles corresponding to an mle for each "batch" or "copy".
    /// Then we refer to the mle that represents the concatenation of these mles by interleaving as the
    /// flattened mle and each individual mle as a batched mle.
    ///
    /// # Arguments
    /// * `nonzero_gates`: the gate wiring between single-copy circuit (as the wiring for each circuit remains the same)
    /// x is the label on the batched mle `lhs`, y is the label on the batched mle `rhs`, and z is the label on the next layer, batched
    /// * `lhs`: the flattened mle representing the left side of the summation
    /// * `rhs`: the flattened mle representing the right side of the summation
    /// * `num_dataparallel_bits`: the number of bits representing the circuit copy we are looking at
    /// * `gate_operation`: which operation the gate is performing. right now, can either be an 'add' or 'mul' gate
    ///
    /// # Returns
    /// A flattened `DenseMle` that represents the evaluations of the add gate wiring on `lhs` and `rhs` over the boolean hypercube
    pub fn add_gate(
        &mut self,
        nonzero_gates: Vec<(usize, usize, usize)>,
        lhs: DenseMleRef<F>,
        rhs: DenseMleRef<F>,
        num_dataparallel_bits: Option<usize>,
        gate_operation: BinaryOperation,
    ) -> DenseMle<F, F>
    where
        T: From<Gate<F>>,
    {
        let id = LayerId::Layer(self.layers.len());
        // constructor for batched mul gate struct
        let gate: Gate<F> = Gate::new(
            num_dataparallel_bits,
            nonzero_gates.clone(),
            lhs.clone(),
            rhs.clone(),
            gate_operation,
            id,
        );
        let max_gate_val = nonzero_gates
            .clone()
            .into_iter()
            .fold(0, |acc, (z, _, _)| std::cmp::max(acc, z));

        // number of entries in the resulting table is the max gate z value * 2 to the power of the number of dataparallel bits, as we are
        // evaluating over all values in the boolean hypercube which includes dataparallel bits
        let num_dataparallel_vals = 1 << (num_dataparallel_bits.unwrap_or(0));
        let res_table_num_entries = (max_gate_val + 1) * num_dataparallel_vals;
        self.layers.push(gate.into());

        // iterate through each of the indices and perform the binary operation specified
        let mut res_table = vec![F::ZERO; res_table_num_entries];
        (0..num_dataparallel_vals).for_each(|idx| {
            nonzero_gates
                .clone()
                .into_iter()
                .for_each(|(z_ind, x_ind, y_ind)| {
                    let f2_val = *lhs
                        .bookkeeping_table()
                        .get(idx + (x_ind * num_dataparallel_vals))
                        .unwrap_or(&F::ZERO);
                    let f3_val = *rhs
                        .bookkeeping_table()
                        .get(idx + (y_ind * num_dataparallel_vals))
                        .unwrap_or(&F::ZERO);
                    res_table[idx + (z_ind * num_dataparallel_vals)] =
                        gate_operation.perform_operation(f2_val, f3_val);
                });
        });

        let res_mle: DenseMle<F, F> = DenseMle::new_from_raw(res_table, id, None);

        res_mle
    }

    /// Creates a new Layers
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Returns the number of layers in the GKR circuit
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<F: FieldExt, T: Layer<F>> Default for Layers<F, T> {
    fn default() -> Self {
        Self::new()
    }
}

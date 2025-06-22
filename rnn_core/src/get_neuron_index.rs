use crate::{structures::ComputedParams, LayerParams};

pub fn get_neuron_index(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    layer_x: usize,
    layer_y: usize,
    neuron_in_field_x: usize,
    neuron_in_field_y: usize,
) -> usize {
    let layer_offset = computed_params.row_size * layer_y + computed_params.field_size * layer_x;
    let field_offset = layer_params.field_width * neuron_in_field_y + neuron_in_field_x;

    layer_offset + field_offset
}

use crate::{structures::ComputedParams, LayerParams};

pub fn get_neuron_coordinates_by_index(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    neuron_index: usize,
) -> (usize, usize) {
    let layer_field_index = neuron_index / computed_params.field_size;
    let neuron_in_field_index = neuron_index % computed_params.field_size;

    let layer_x = layer_field_index % layer_params.layer_width;
    let layer_y = (layer_field_index - layer_x) / layer_params.layer_height;

    let layer_x_offset = layer_params.field_width * layer_x;
    let layer_y_offset = layer_params.field_height * layer_y;

    let field_x_offset = neuron_in_field_index % layer_params.field_width;
    let field_y_offset = neuron_in_field_index / layer_params.field_width;

    (
        layer_x_offset + field_x_offset,
        layer_y_offset + field_y_offset,
    )
}

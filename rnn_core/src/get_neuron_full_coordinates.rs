use crate::LayerParams;

pub fn get_neuron_full_coordinates(
    params: &LayerParams,
    neuron_x: usize,
    neuron_y: usize,
) -> (usize, usize, usize, usize) {
    let neuron_in_field_x = neuron_x % params.field_width;
    let neuron_in_field_y = neuron_y % params.field_height;

    let layer_x = (neuron_x - neuron_in_field_x) / params.field_width;
    let layer_y = (neuron_y - neuron_in_field_y) / params.field_height;

    (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y)
}

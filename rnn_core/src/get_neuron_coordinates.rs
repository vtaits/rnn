use crate::LayerParams;

pub fn get_neuron_coordinates(
    params: &LayerParams,
    layer_x: usize,
    layer_y: usize,
    neuron_in_field_x: usize,
    neuron_in_field_y: usize,
) -> (usize, usize) {
    let layer_x_offset = params.field_width * layer_x;
    let layer_y_offset = params.field_height * layer_y;

    let field_x_offset = neuron_in_field_x;
    let field_y_offset = neuron_in_field_y;

    (
        layer_x_offset + field_x_offset,
        layer_y_offset + field_y_offset,
    )
}

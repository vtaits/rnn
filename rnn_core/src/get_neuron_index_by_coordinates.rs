use crate::{
    get_neuron_full_coordinates::get_neuron_full_coordinates, get_neuron_index::get_neuron_index,
    structures::ComputedParams, LayerParams,
};

pub fn get_neuron_index_by_coordinates(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    neuron_x: usize,
    neuron_y: usize,
) -> usize {
    let (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y) =
        get_neuron_full_coordinates(layer_params, neuron_x, neuron_y);

    get_neuron_index(
        layer_params,
        computed_params,
        layer_x,
        layer_y,
        neuron_in_field_x,
        neuron_in_field_y,
    )
}

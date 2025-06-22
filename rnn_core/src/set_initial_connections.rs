use ndarray::Array2;

use crate::{
    get_neuron_coordinates::get_neuron_coordinates,
    get_neuron_index::get_neuron_index,
    get_neuron_index_by_coordinates::get_neuron_index_by_coordinates,
    spiral::{get_last_field, get_next_field},
    structures::{ComputedParams, SynapseMask},
    LayerParams, SynapseParams,
};

fn apply_mask(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    base_neuron_index: usize,
    distance_weights: &mut Array2<f32>,
    mask: &SynapseMask,
    x: usize,
    y: usize,
) {
    for iter_x in 0..mask.size {
        let offset_x = iter_x as i32 - mask.offset as i32;
        let neuron_x = (x as i32) + offset_x;

        if neuron_x < 0 || (neuron_x as usize) > computed_params.row_width - 1 {
            continue;
        }

        for iter_y in 0..mask.size {
            let offset_y = iter_y as i32 - mask.offset as i32;
            let neuron_y = (y as i32) + offset_y;

            if neuron_y < 0 || (neuron_y as usize) > computed_params.column_height - 1 {
                continue;
            }

            let target_neuron_index = get_neuron_index_by_coordinates(
                layer_params,
                computed_params,
                neuron_x as usize,
                neuron_y as usize,
            );

            let value = mask.mask[[iter_x, iter_y]];

            distance_weights[[target_neuron_index, base_neuron_index]] = value;
        }
    }
}

pub fn set_initial_connections(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
    mask: &SynapseMask,
) -> (
    // distance_weights of synapses from the first layer to the second layer
    Array2<f32>,
    // distance_weights of synapses from the second layer to the first layer
    Array2<f32>,
    // accumulated of synapses from the first layer to the second layer
    Array2<f32>,
    // accumulated of synapses from the second layer to the first layer
    Array2<f32>,
) {
    let layer_size = layer_params.field_width
        * layer_params.field_height
        * layer_params.layer_width
        * layer_params.layer_height;

    let mut distance_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut distance_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let mut accumulated_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut accumulated_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let (finish_x, finish_y) = get_last_field(layer_params);

    for layer_y in 0..layer_params.layer_height {
        for layer_x in 0..layer_params.layer_width {
            let (layer_2_to_1_x, layer_2_to_1_y) = get_next_field(layer_params, layer_x, layer_y);

            for neuron_in_field_y in 0..layer_params.field_height {
                for neuron_in_field_x in 0..layer_params.field_width {
                    // from 1 to 2
                    let neuron_index = get_neuron_index(
                        layer_params,
                        computed_params,
                        layer_x,
                        layer_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    accumulated_weights_1_to_2[[neuron_index, neuron_index]] =
                        synapse_params.initial_strong_g;

                    let (x, y) = get_neuron_coordinates(
                        layer_params,
                        layer_x,
                        layer_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    apply_mask(
                        layer_params,
                        computed_params,
                        neuron_index,
                        &mut distance_weights_1_to_2,
                        mask,
                        x,
                        y,
                    );

                    // the last field have no connection to the first layer
                    if layer_x != finish_x || layer_y != finish_y {
                        // from 2 to 1
                        let neuron_2_to_1_index = get_neuron_index(
                            layer_params,
                            computed_params,
                            layer_2_to_1_x,
                            layer_2_to_1_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        accumulated_weights_2_to_1[[neuron_2_to_1_index, neuron_index]] =
                            synapse_params.initial_strong_g;

                        let (x_2_to_1, y_2_to_1) = get_neuron_coordinates(
                            layer_params,
                            layer_2_to_1_x,
                            layer_2_to_1_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        apply_mask(
                            layer_params,
                            computed_params,
                            neuron_index,
                            &mut distance_weights_2_to_1,
                            mask,
                            x_2_to_1,
                            y_2_to_1,
                        );
                    }
                }
            }
        }
    }

    (
        distance_weights_1_to_2,
        distance_weights_2_to_1,
        accumulated_weights_1_to_2,
        accumulated_weights_2_to_1,
    )
}

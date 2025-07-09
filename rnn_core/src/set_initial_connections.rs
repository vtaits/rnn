use ndarray::Array2;

use crate::{
    get_neuron_coordinates::get_neuron_coordinates,
    get_neuron_index::get_neuron_index,
    get_neuron_index_by_coordinates::get_neuron_index_by_coordinates,
    spiral::{get_last_field, get_next_field},
    structures::{ComputedParams, InitialConnections, SynapseMask},
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
    has_connections: &Array2<bool>,
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

            if has_connections[[target_neuron_index, base_neuron_index]] {
                distance_weights[[target_neuron_index, base_neuron_index]] = value;
            }
        }
    }
}

fn fill_conntected_fields(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    has_connections: &mut Array2<bool>,
    layer_from_x: usize,
    layer_from_y: usize,
    layer_to_x: usize,
    layer_to_y: usize,
) {
    for neuron_in_field_from_y in 0..layer_params.field_height {
        for neuron_in_field_from_x in 0..layer_params.field_width {
            let neuron_from_index = get_neuron_index(
                layer_params,
                computed_params,
                layer_from_x,
                layer_from_y,
                neuron_in_field_from_x,
                neuron_in_field_from_y,
            );

            for neuron_in_field_to_y in 0..layer_params.field_height {
                for neuron_in_field_to_x in 0..layer_params.field_width {
                    let neuron_to_index = get_neuron_index(
                        layer_params,
                        computed_params,
                        layer_to_x,
                        layer_to_y,
                        neuron_in_field_to_x,
                        neuron_in_field_to_y,
                    );

                    has_connections[[neuron_to_index, neuron_from_index]] = true;
                }
            }
        }
    }
}

fn fill_has_conntections(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    has_connections_1_to_2: &mut Array2<bool>,
    has_connections_2_to_1: &mut Array2<bool>,
) {
    let (last_layer_x, last_layer_y) = get_last_field(layer_params);

    let mut prev_fields = vec![];

    let mut cur_layer_x = 0usize;
    let mut cur_layer_y = 0usize;

    loop {
        for (prev_layer_x, prev_layer_y) in prev_fields.iter() {
            fill_conntected_fields(
                layer_params,
                computed_params,
                has_connections_1_to_2,
                cur_layer_x,
                cur_layer_y,
                *prev_layer_x,
                *prev_layer_y,
            );

            fill_conntected_fields(
                layer_params,
                computed_params,
                has_connections_2_to_1,
                cur_layer_x,
                cur_layer_y,
                *prev_layer_x,
                *prev_layer_y,
            );
        }

        fill_conntected_fields(
            layer_params,
            computed_params,
            has_connections_1_to_2,
            cur_layer_x,
            cur_layer_y,
            cur_layer_x,
            cur_layer_y,
        );

        // fill_conntected_fields(
        //     layer_params,
        //     computed_params,
        //     has_connections_2_to_1,
        //     cur_layer_x,
        //     cur_layer_y,
        //     cur_layer_x,
        //     cur_layer_y,
        // );

        if cur_layer_x == last_layer_x && cur_layer_y == last_layer_y {
            return;
        }

        let (next_field_x, next_field_y) = get_next_field(layer_params, cur_layer_x, cur_layer_y);

        fill_conntected_fields(
            layer_params,
            computed_params,
            has_connections_2_to_1,
            cur_layer_x,
            cur_layer_y,
            next_field_x,
            next_field_y,
        );

        if cur_layer_x == 0 && cur_layer_y == 0 {
            //    prev_fields.push((cur_layer_x, cur_layer_y));
        }

        cur_layer_x = next_field_x;
        cur_layer_y = next_field_y;
    }
}

pub fn set_initial_connections(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
    mask: &SynapseMask,
) -> InitialConnections {
    let layer_size = layer_params.field_width
        * layer_params.field_height
        * layer_params.layer_width
        * layer_params.layer_height;

    let mut distance_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut distance_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    // synapses to identical map from the first layer to the second layer
    let mut strong_synapses_1_to_2 = vec![0u64; layer_size];
    // synapses to identical map from the second layer to the first layer
    let mut strong_synapses_2_to_1 = vec![0u64; layer_size];

    let mut accumulated_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut accumulated_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let mut has_conntections_1_to_2 = Array2::<bool>::default([layer_size, layer_size]);
    let mut has_conntections_2_to_1 = Array2::<bool>::default([layer_size, layer_size]);

    fill_has_conntections(
        layer_params,
        computed_params,
        &mut has_conntections_1_to_2,
        &mut has_conntections_2_to_1,
    );

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

                    accumulated_weights_1_to_2[[neuron_index, neuron_index]] = synapse_params.max_g;

                    strong_synapses_1_to_2[neuron_index] = neuron_index as u64;

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
                        &has_conntections_1_to_2,
                    );

                    if layer_x == 0 && layer_y == 0 {
                        strong_synapses_2_to_1[neuron_index] = layer_size as u64;
                    }

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
                            synapse_params.max_g;

                        strong_synapses_2_to_1[neuron_2_to_1_index] = neuron_index as u64;

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
                            &has_conntections_2_to_1,
                        );
                    }
                }
            }
        }
    }

    InitialConnections {
        distance_weights_1_to_2,
        distance_weights_2_to_1,
        strong_synapses_1_to_2,
        strong_synapses_2_to_1,
        accumulated_weights_1_to_2,
        accumulated_weights_2_to_1,
    }
}

use crate::excite_neurons_with_partitions::excite_neurons_with_partitions;
use crate::excite_neurons_without_partitions::excite_neurons_without_partitions;
use crate::structures::ComputedParams;
use crate::{LayerParams, SynapseParams};

fn get_weight_coefficient(
    gamma_inc: &f32,
    gamma_dec: &f32,
    accumulated_weight: &f32,
    g_0: &f32,
) -> f32 {
    let shifted_accumulated_weight = accumulated_weight - g_0;

    if shifted_accumulated_weight < 0.0 {
        return (-gamma_dec * shifted_accumulated_weight).exp() - 1.0;
    }

    return 1.0 - (-gamma_inc * shifted_accumulated_weight).exp();
}

fn remove_extra_neurons(neurons: &mut Vec<u8>, limit: usize) {
    let excited_count = neurons.iter().filter(|&&x| x != 0).count();

    if excited_count <= limit {
        return;
    }

    let mut neurons_to_remove = excited_count - limit;

    while neurons_to_remove > 0 {
        let index = rand::random_range(0..neurons.iter().count());

        if neurons[index] > 0 {
            neurons[index] = 0;
            neurons_to_remove -= 1;
        }
    }
}

/**
 * Only to count benchmarks
 */
pub fn apply_synapses_cpu_fallback(
    is_prediction: bool,
    layer_size: usize,
    forward_synapses: &mut Vec<f32>,
    forward_distance_weights: &Vec<f32>,
    restore_distance_weights: Option<&Vec<f32>>,
    offsets: &Vec<u64>,
    restore_offsets: Option<&Vec<u64>>,
    mut restore_synapses: Option<&mut Vec<f32>>,
    restore_offsets_count: &usize,
    neurons_from: &Vec<u8>,
    neurons_to: &mut Vec<u8>,
    refract_intervals_to: &Vec<u8>,
    layer_params: &LayerParams,
    synapse_params: &SynapseParams,
    computed_params: &ComputedParams,
    threshold: f32,
    layer_index: usize,
    signals_to: &mut Vec<f32>,
) -> ocl::Result<()> {
    let LayerParams { partitions, .. } = layer_params;

    let SynapseParams {
        gamma_inc,
        gamma_dec,
        g_0,
        g_dec,
        g_inc,
        min_g,
        max_g,
        ..
    } = synapse_params;

    let ComputedParams {
        field_size: field_size_ref,
        field_count,
        excited_neurons_limit,
        ..
    } = computed_params;

    let g_0_for_layer = if layer_index == 2 { g_0 } else { &0.0 };
    let field_size = *field_size_ref;

    // CPU realization of `apply_synapses.cl`

    for row in 0..layer_size {
        let neuron_in_field_x = row % field_size;
        let offset = offsets[row] as usize;

        if refract_intervals_to[row] > 0 {
            if is_prediction {
                signals_to[row] = 0.0;
            } else {
                neurons_to[row] = 0;
            }
        } else if !is_prediction {
            let index_from = offset + neuron_in_field_x;

            if index_from < layer_size {
                neurons_to[row] = neurons_from[index_from];
            }
        } else {
            let mut sum = 0.0;

            if offset < layer_size {
                for col in 0..field_size {
                    let neuron_from_index = offset + col;

                    if neurons_from[neuron_from_index] > 0 {
                        let weight_to = forward_synapses[col * layer_size + row];

                        if weight_to > 0.0001 || weight_to < -0.0001 {
                            sum += get_weight_coefficient(
                                gamma_inc,
                                gamma_dec,
                                &weight_to,
                                g_0_for_layer,
                            ) * forward_distance_weights
                                [col * field_size + neuron_in_field_x];
                        }
                    }
                }
            }

            if row < field_size {
                match restore_offsets {
                    Some(restore_offsets) => match restore_distance_weights {
                        Some(restore_distance_weights) => match &restore_synapses {
                            Some(restore_synapses) => {
                                for restore_offset_index in 0..*restore_offsets_count {
                                    for col in 0..field_size {
                                        let neuron_from_index =
                                            restore_offsets[restore_offset_index] as usize + col;

                                        if neurons_from[neuron_from_index] > 0 {
                                            let weight_to = restore_synapses[restore_offset_index
                                                * field_size
                                                * field_size
                                                + row * field_size
                                                + col];

                                            if weight_to > 0.0001 || weight_to < -0.0001 {
                                                sum += get_weight_coefficient(
                                                    gamma_inc, gamma_dec, &weight_to, g_0,
                                                ) * restore_distance_weights
                                                    [restore_offset_index
                                                        * field_size
                                                        * field_size
                                                        + neuron_in_field_x * field_size
                                                        + col];
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }

            signals_to[row] = sum;
        }

        if !is_prediction {
            // recount synapses
            if offset < layer_size {
                for col in 0..field_size {
                    let neuron_from_index = offset + col;
                    let synapse_index = col * layer_size + row;

                    if neurons_from[neuron_from_index] > 0 {
                        if forward_distance_weights[col * field_size + neuron_in_field_x] < 0.001 {
                            continue;
                        }

                        let prev_value = forward_synapses[synapse_index];

                        if refract_intervals_to[row] > 0 {
                            if forward_synapses[synapse_index] > *min_g {
                                forward_synapses[synapse_index] = (prev_value - g_dec).max(0.0);
                            }
                        } else if neurons_to[row] > 0 {
                            if forward_synapses[synapse_index] < *max_g {
                                forward_synapses[synapse_index] = (prev_value + g_inc).min(*max_g);
                            }
                        }
                    }
                }
            }

            if row < field_size {
                match restore_offsets {
                    Some(restore_offsets) => match restore_distance_weights {
                        Some(restore_distance_weights) => match restore_synapses.as_mut() {
                            Some(restore_synapses) => {
                                for restore_offset_index in 0..*restore_offsets_count {
                                    for col in 0..field_size {
                                        let neuron_from_index =
                                            restore_offsets[restore_offset_index] as usize + col;
                                        let synapse_index =
                                            restore_offset_index * field_size * field_size
                                                + row * field_size
                                                + col;

                                        if neurons_from[neuron_from_index] > 0 {
                                            if restore_distance_weights[restore_offset_index
                                                * field_size
                                                * field_size
                                                + neuron_in_field_x * field_size
                                                + col]
                                                < 0.001
                                            {
                                                continue;
                                            }

                                            let prev_value = restore_synapses[synapse_index];

                                            if refract_intervals_to[row] > 0 {
                                                if restore_synapses[synapse_index] > *min_g {
                                                    restore_synapses[synapse_index] =
                                                        (prev_value - g_dec).max(0.0);
                                                }
                                            } else if neurons_to[row] > 0 {
                                                if restore_synapses[synapse_index] < *max_g {
                                                    restore_synapses[synapse_index] =
                                                        (prev_value + g_inc).min(*max_g);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    if is_prediction {
        if let Some(partitions) = partitions {
            excite_neurons_with_partitions(
                neurons_to,
                &signals_to,
                field_size,
                *field_count,
                threshold,
                partitions,
            );
        } else {
            excite_neurons_without_partitions(neurons_to, &signals_to, threshold);

            remove_extra_neurons(neurons_to, *excited_neurons_limit);
        }
    }

    Ok(())
}

float get_weight_coefficient(
    const float gamma_inc,
    const float gamma_dec,
    const float accumulated_weight,
    const float g_0
) {
    float shifted_accumulated_weight = accumulated_weight - g_0;

    if (shifted_accumulated_weight < 0) {
        return exp(-gamma_dec * shifted_accumulated_weight) - 1.0;
    }

    return 1.0 - exp(-gamma_inc * shifted_accumulated_weight);
}

__kernel void apply_synapses(
    __global float* forward_synapses,
    __global float* forward_distance_weights,
    __global float* restore_distance_weights,
    __global ulong* offsets,
    __global ulong* restore_offsets,
    const unsigned int restore_offsets_count,
    __global float* restore_synapses,
    __global unsigned char* neurons_from,
    __global unsigned char* refract_intervals_to,
    __global unsigned char* neurons_to,
    __global float* signals_to,
    const unsigned int layer_size,
    const unsigned int field_size,
    const float threshold,
    const float gamma_inc,
    const float gamma_dec,
    const float g_0,
    const float g_dec,
    const float g_inc,
    const float min_g,
    const float max_g,
    const uchar is_prediction,
    __global int* inc_counter,
    __global int* dec_counter
) {
    // recount neurons
    int row = get_global_id(0);
    int neuron_in_field_x = row % field_size;

    if (refract_intervals_to[row] > 0) {
        if (is_prediction) {
          signals_to[row] = 0;
        } else {
          neurons_to[row] = 0;
        }
    } else if (!is_prediction) {
        ulong index_from = offsets[row] + neuron_in_field_x;

        if (index_from < layer_size) {
            neurons_to[row] = neurons_from[index_from];
        }
    } else {
        float sum = 0.0;

        if (offsets[row] < layer_size) {
            for (uint col = 0; col < field_size; ++col) {
                unsigned int neuron_from_index = offsets[row] + col;

                if (neurons_from[neuron_from_index] > 0) {
                    float weight_to = forward_synapses[col * layer_size + row];

                    if (weight_to > 0.0001 || weight_to < -0.0001) {
                        sum += get_weight_coefficient(gamma_inc, gamma_dec, weight_to, g_0) * forward_distance_weights[col * field_size + neuron_in_field_x];
                    }
                }
            }
        }

        if (row < field_size) {
            for (unsigned int restore_offset_index = 0; restore_offset_index < restore_offsets_count; ++restore_offset_index) {
                for (uint col = 0; col < field_size; ++col) {
                    unsigned int neuron_from_index = restore_offsets[restore_offset_index] + col;

                    if (neurons_from[neuron_from_index] > 0) {
                        float weight_to = restore_synapses[restore_offset_index * field_size * field_size + row * field_size + col];

                        if (weight_to > 0.0001 || weight_to < -0.0001) {
                            sum += get_weight_coefficient(gamma_inc, gamma_dec, weight_to, g_0) * restore_distance_weights[restore_offset_index * field_size * field_size + neuron_in_field_x * field_size + col];
                        }
                    }
                }
            }
        }

        signals_to[row] = sum;
    }

    if (is_prediction) {
        return;
    }

    // recount synapses
    if (offsets[row] < layer_size) {
        for (uint col = 0; col < field_size; ++col) {
            unsigned int neuron_from_index = offsets[row] + col;
            unsigned int synapse_index = col * layer_size + row;

            if (neurons_from[neuron_from_index] > 0) {
                if (forward_distance_weights[col * field_size + neuron_in_field_x] < 0.001) {
                    continue;
                }

                float prev_value = forward_synapses[synapse_index];

                if (refract_intervals_to[row] > 0) {
                    if (forward_synapses[synapse_index] > min_g) {
                        forward_synapses[synapse_index] = max(prev_value - g_dec, 0.0f);

                        // atomic_inc(&dec_counter[0]);
                        // #ifdef DEBUG
                            // atomic_inc(*dec_counter[0]);
                        // #endif
                    }
                } else if (neurons_to[row] > 0) {
                    if (forward_synapses[synapse_index] < max_g) {
                        forward_synapses[synapse_index] = min(prev_value + g_inc, max_g);

                        // atomic_inc(&inc_counter[0]);
                        // #ifdef DEBUG
                        //    atomic_inc(&inc_counter[0]);
                        // #endif
                    }
                }
            }
        }
    }

    if (row < field_size) {
        for (unsigned int restore_offset_index = 0; restore_offset_index < restore_offsets_count; ++restore_offset_index) {
            for (uint col = 0; col < field_size; ++col) {
                unsigned int neuron_from_index = restore_offsets[restore_offset_index] + col;
                unsigned int synapse_index = restore_offset_index * field_size * field_size + row * field_size + col;

                if (neurons_from[neuron_from_index] > 0) {
                    if (restore_distance_weights[restore_offset_index * field_size * field_size + neuron_in_field_x * field_size + col] < 0.001) {
                        continue;
                    }

                    float prev_value = restore_synapses[synapse_index];

                    if (refract_intervals_to[row] > 0) {
                        if (restore_synapses[synapse_index] > min_g) {
                            restore_synapses[synapse_index] = max(prev_value - g_dec, 0.0f);

                            // atomic_inc(&dec_counter[0]);
                            // #ifdef DEBUG
                                // atomic_inc(*dec_counter[0]);
                            // #endif
                        }
                    } else if (neurons_to[row] > 0) {
                        if (restore_synapses[synapse_index] < max_g) {
                            restore_synapses[synapse_index] = min(prev_value + g_inc, max_g);

                            // atomic_inc(&inc_counter[0]);
                            // #ifdef DEBUG
                            //    atomic_inc(&inc_counter[0]);
                            // #endif
                        }
                    }
                }
            }
        }
    }
}

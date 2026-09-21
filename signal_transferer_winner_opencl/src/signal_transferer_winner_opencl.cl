float get_weight_coefficient(
    const float gamma_inc,
    const float gamma_dec,
    const float accumulated_weight,
    const float g_0
) {
    float shifted_accumulated_weight = accumulated_weight - g_0;

    if (shifted_accumulated_weight < 0) {
        return exp(gamma_dec * shifted_accumulated_weight) - 1.0;
    }

    return 1.0 - exp(-gamma_inc * shifted_accumulated_weight);
}

__kernel void signal_transferer_winner_opencl(
    const float g_0,
    __global unsigned char* neurons_from,
    __global float* signals_to,
    __global unsigned char* refract_intervals_to,
    __global float* synapses,
    __global float* distances,
    const unsigned int layer_size,
    const float gamma_inc,
    const float gamma_dec
) {
    int neuron_to_index = get_global_id(0);

    signals_to[neuron_to_index] = 0.0;

    if (refract_intervals_to[neuron_to_index] == 0) {
        for (uint neuron_from_index = 0; neuron_from_index < layer_size; ++neuron_from_index) {
            uint synapse_index = neuron_from_index * layer_size + neuron_to_index;

            if (neurons_from[neuron_from_index]) {
                float weight_to = synapses[synapse_index];

                if (weight_to > 0.0001 || weight_to < -0.0001) {
                    signals_to[neuron_to_index] += get_weight_coefficient(
                        gamma_inc,
                        gamma_dec,
                        weight_to,
                        g_0
                    ) * distances[synapse_index];
                }
            }
        }
    }
}

__kernel void signal_transferer_strong_opencl(
    const float g_0,
    __global unsigned char* neurons_from,
    __global unsigned char* neurons_to,
    __global unsigned char* refract_intervals_to,
    __global float* synapses,
    __global float* distances,
    const unsigned int layer_size,
    const float threshold,
    const float gamma_inc,
    const float gamma_dec,
    const float g_dec,
    const float g_inc,
    const float min_g,
    const float max_g
) {
    int neuron_to_index = get_global_id(0);

    if (refract_intervals_to[neuron_to_index] > 0) {
        neurons_to[neuron_to_index] = 0;
    } else {
        neurons_to[neuron_to_index] = 0;

        for (uint neuron_from_index = 0; neuron_from_index < layer_size; ++neuron_from_index) {
            uint synapse_index = neuron_from_index * layer_size + neuron_to_index;

            if (neurons_from[neuron_from_index] && distances[synapse_index] > 0.999) {
                neurons_to[neuron_to_index] = 1;
            }
        }
    }

    for (uint neuron_from_index = 0; neuron_from_index < layer_size; ++neuron_from_index) {
        if (!neurons_from[neuron_from_index]) {
            continue;
        }

        uint synapse_index = neuron_from_index * layer_size + neuron_to_index;

        if (refract_intervals_to[neuron_to_index] > 0) {
            if (synapses[synapse_index] > min_g) {
                synapses[synapse_index] = max(synapses[synapse_index] - g_dec, min_g);
            }
        } else if (neurons_to[neuron_to_index]) {
            if (synapses[synapse_index] < max_g) {
                synapses[synapse_index] = min(synapses[synapse_index] + g_inc, max_g);
            }
        }
    }
}

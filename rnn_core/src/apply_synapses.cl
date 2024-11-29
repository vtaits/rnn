float get_weight_coefficient(
    const float gamma_inc,
    const float gamma_dec,
    const float accumulated_weight,
    const float g_0
) {
    float shifted_accumulated_weight = accumulated_weight - g_0;

    if (shifted_accumulated_weight < 0) {
        return exp(-gamma_inc * shifted_accumulated_weight) - 1.0;
    }

    return 1.0 - exp(-gamma_inc * shifted_accumulated_weight);
}

__kernel void apply_synapses(
    __global float* accumulated_weights,
    __global float* distance_weights,
    __global unsigned char* neurons_from,
    __global unsigned char* refract_intervals_to,
    __global unsigned char* next_neurons_to,
    const unsigned int layer_size,
    const uchar initial_refract_interval,
    const float threshold,
    const float gamma_inc,
    const float gamma_dec,
    const float g_0
) {
    int row = get_global_id(0);

    if (refract_intervals_to[row] > 0) {
        next_neurons_to[row] = 0;
        return;
    }

    float sum = 0.0;
    for (int col = 0; col < layer_size; ++col) {
        if (neurons_from[col] > 0) {
            float weight_to = accumulated_weights[row * layer_size + col];

            if (weight_to > 0.0001 || weight_to < -0.0001) {
                sum += get_weight_coefficient(gamma_inc, gamma_dec, weight_to, g_0) * distance_weights[row * layer_size + col];

                if (sum > threshold) {
                    next_neurons_to[row] = 1;
                    return;
                }
            }
        }
    }

    next_neurons_to[row] = 0;
}

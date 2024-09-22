float get_weight_coefficient(
    const float gamma,
    const float accumulated_weight,
    const float g_0
) {
    if (accumulated_weight < g_0) {
        return 0.0;
    }

    return 1.0 - exp(-gamma * (accumulated_weight - g_0));
}

__kernel void apply_synapses(
    __global float* accumulated_weights,
    __global float* distance_weights,
    __global float* neurons_from,
    __global unsigned char* refract_intervals_to,
    __global float* next_neurons_to,
    const unsigned int layer_size,
    const uchar initial_refract_interval,
    const float threshold,
    const float gamma,
    const float g_0
) {
    int row = get_global_id(0);

    if (refract_intervals_to[row] > 0) {
        next_neurons_to[row] = 0.0;
        return;
    }

    float sum = 0.0;
    for (int col = 0; col < layer_size; ++col) {
        sum += get_weight_coefficient(gamma, accumulated_weights[row * layer_size + col], g_0) * distance_weights[row * layer_size + col] * neurons_from[col];

        if (sum > threshold) {
            next_neurons_to[row] = 1.0;
            return;
        }
    }

    next_neurons_to[row] = 0.0;
}

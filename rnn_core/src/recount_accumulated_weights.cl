__kernel void recount_accumulated_weights(
    __global float* accumulated_weights,
    __global float* neurons_from,
    __global float* neurons_to,
    __global unsigned char* refract_intervals_to,
    __global float* next_accumulated_weights,
    const unsigned int layer_size,
    const float g_dec,
    const float g_inc
) {
    int row = get_global_id(0);

    if (neurons_from[row] < 0.5) {
        for (int col = 0; col < layer_size; ++col) {
            unsigned int index_to = row * layer_size + col;
            next_accumulated_weights[index_to] = accumulated_weights[index_to];
        }

        return;
    }

    for (int col = 0; col < layer_size; ++col) {
        unsigned int index_to = row * layer_size + col;

        if (refract_intervals_to[index_to] > 0) {
            next_accumulated_weights[index_to] = accumulated_weights[index_to] - g_dec;
        } else if (neurons_from[row] > 0.5) {
            next_accumulated_weights[index_to] = accumulated_weights[index_to] + g_inc;
        } else {
            next_accumulated_weights[index_to] = accumulated_weights[index_to];
        }
    }
}

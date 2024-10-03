__kernel void recount_accumulated_weights(
    __global float* accumulated_weights,
    __global float* neurons_from,
    __global float* neurons_to,
    __global unsigned char* refract_intervals_to,
    __global float* next_accumulated_weights,
    const unsigned int layer_size,
    const float g_dec,
    const float g_inc,
    const float max_g
) {
    int row = get_global_id(0);

    for (int col = 0; col < layer_size; ++col) {
        unsigned int index_to = row * layer_size + col;

        if (neurons_from[col] < 0.5) {
            next_accumulated_weights[index_to] = accumulated_weights[index_to];
        } else if (refract_intervals_to[index_to] > 0) {
            next_accumulated_weights[index_to] = max(accumulated_weights[index_to] - g_dec, 0.0f);
        } else if (neurons_to[row] > 0.5) {
            next_accumulated_weights[index_to] = min(accumulated_weights[index_to] + g_inc, max_g);
        } else  {
            next_accumulated_weights[index_to] = accumulated_weights[index_to];
        }
    }
}

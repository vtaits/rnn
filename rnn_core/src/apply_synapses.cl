__kernel void apply_synapses(
    __global float* synapses,
    __global float* neurons,
    __global float* result,
    const unsigned int layer_size,
    const float threshold
) {
    int row = get_global_id(0);
    float sum = 0.0;
    for (int col = 0; col < layer_size; ++col) {
        sum += synapses[row * layer_size + col] * neurons[col];

        if (sum > threshold) {
            result[row] = 1.0;
            return;
        }
    }

    result[row] = 0.0;
}

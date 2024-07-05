__kernel void apply_synapses(
    __global float* synapses,
    __global float* neurons,
    __global unsigned char* refract_intervals,
    __global float* result,
    const unsigned int layer_size,
    const uchar initial_refract_interval,
    const float threshold
) {
    int row = get_global_id(0);

    if (refract_intervals[row] > 0) {
        result[row] = 1.0;
        refract_intervals[row] = refract_intervals[row] - 1;
        return;
    }

    float sum = 0.0;
    for (int col = 0; col < layer_size; ++col) {
        sum += synapses[row * layer_size + col] * neurons[col];

        if (sum > threshold) {
            result[row] = 1.0;
            refract_intervals[row] = initial_refract_interval;
            return;
        }
    }

    result[row] = 0.0;
}

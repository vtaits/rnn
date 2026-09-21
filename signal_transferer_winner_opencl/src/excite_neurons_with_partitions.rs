fn apply_partition(
    neurons: &mut [u8],
    signals: &[f32],
    threshold: &f32,
    start_partition_index: usize,
    partition_size: &usize,
) {
    let mut partition_res: (usize, f32) = (start_partition_index, signals[start_partition_index]);

    for neuron_index in start_partition_index..start_partition_index + partition_size {
        let signal_value = signals[neuron_index];

        neurons[neuron_index] = 0;

        if signal_value > partition_res.1 {
            partition_res = (neuron_index, signal_value);
        }
    }

    if partition_res.1 > *threshold {
        neurons[partition_res.0] = 1;
    }
}

pub fn excite_neurons_with_partitions(
    neurons: &mut [u8],
    signals: &[f32],
    field_size: &usize,
    field_count: &usize,
    threshold: &f32,
    partitions: &[usize],
) {
    for field_index in 0..*field_count {
        let mut start_partition_index = field_index * field_size;

        for partition in partitions {
            apply_partition(
                neurons,
                signals,
                threshold,
                start_partition_index,
                partition,
            );

            start_partition_index += partition;
        }
    }
}

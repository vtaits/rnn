use crate::structures::Partition;

fn apply_partition(
    neurons: &mut Vec<u8>,
    signals: &Vec<f32>,
    threshold: f32,
    start_partition_index: usize,
    full_partition_size: usize,
    accept_all: bool,
) {
    let mut partition_res: (usize, f32) = (start_partition_index, signals[start_partition_index]);

    for neuron_index in start_partition_index + 1..start_partition_index + full_partition_size {
        let signal_value = signals[neuron_index];
        let is_more_than_threshold = signal_value > threshold;

        if accept_all {
            neurons[neuron_index] = if is_more_than_threshold { 1 } else { 0 };
            continue;
        }

        neurons[neuron_index] = 0;

        if is_more_than_threshold && signal_value > partition_res.1 {
            partition_res = (neuron_index, signal_value);
        }
    }

    if partition_res.1 > threshold {
        neurons[partition_res.0] = 1;
    }
}

pub fn excite_neurons_with_partitions(
    neurons: &mut Vec<u8>,
    signals: &Vec<f32>,
    field_size: usize,
    field_count: usize,
    threshold: f32,
    partitions: &Vec<Partition>,
) {
    for field_index in 0..field_count {
        let mut start_partition_index = field_index * field_size;

        for partition in partitions {
            let full_partition_size = partition.size * 2;

            apply_partition(
                neurons,
                signals,
                threshold,
                start_partition_index,
                full_partition_size,
                partition.accept_all,
            );

            start_partition_index += full_partition_size;
        }
    }
}

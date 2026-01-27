use crate::{
    get_neuron_index::get_neuron_index,
    spiral::{get_last_field, get_next_field},
    structures::{ComputedParams, InitialConnections, PartitionPayloadByIndex, SynapseMask},
    LayerParams, SynapseParams,
};

fn apply_mask_to_block(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    distance_weights: &mut Vec<f32>,
    mask: &SynapseMask,
    offset: usize,
    x: usize,
    y: usize,
    has_connections: &Vec<bool>,
) {
    let base_neuron_index =
        (y % layer_params.field_height) * layer_params.field_width + x % layer_params.field_width;

    for iter_x in 0..mask.size {
        let offset_x = iter_x as i32 - mask.offset as i32;
        let neuron_x = (x as i32) + offset_x;

        if neuron_x < 0 || (neuron_x as usize) > layer_params.field_width - 1 {
            continue;
        }

        for iter_y in 0..mask.size {
            let offset_y = iter_y as i32 - mask.offset as i32;
            let neuron_y = (y as i32) + offset_y;

            if neuron_y < 0 || (neuron_y as usize) > layer_params.field_height - 1 {
                continue;
            }

            let target_neuron_index =
                (neuron_y as usize) * layer_params.field_width + (neuron_x as usize);

            let value = mask.mask[iter_x + iter_y * mask.size];

            if has_connections[target_neuron_index * computed_params.field_size + base_neuron_index]
            {
                distance_weights[offset
                    * computed_params.field_size
                    * computed_params.field_size
                    + target_neuron_index * computed_params.field_size
                    + base_neuron_index] = value;
            }
        }
    }
}

fn check_has_connections_by_partitions(
    partition_from: Option<&PartitionPayloadByIndex>,
    partition_to: Option<&PartitionPayloadByIndex>,
) -> bool {
    match partition_to {
        Some(partition_to_data) => {
            if partition_to_data.correlate_only_self {
                let result = match partition_from {
                    Some(partition_from_data) => {
                        partition_from_data.partition_index == partition_to_data.partition_index
                    }
                    _ => false,
                };

                return result;
            }

            if let Some(no_correlate) = &partition_to_data.no_correlate {
                if let Some(partition_from_data) = partition_from {
                    if no_correlate.contains(&partition_from_data.partition_index) {
                        return false;
                    }
                }
            }

            if let Some(correlate_only) = &partition_to_data.correlate_only {
                let result = match partition_from {
                    Some(partition_from_data) => {
                        correlate_only.contains(&partition_from_data.partition_index)
                    }
                    _ => false,
                };

                return result;
            }

            true
        }
        _ => true,
    }
}

fn fill_conntected_fields(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    has_connections: &mut Vec<bool>,
    layer_from_x: usize,
    layer_from_y: usize,
    layer_to_x: usize,
    layer_to_y: usize,
) {
    for neuron_in_field_from_y in 0..layer_params.field_height {
        for neuron_in_field_from_x in 0..layer_params.field_width {
            let index_in_field_from =
                layer_params.field_width * neuron_in_field_from_y + neuron_in_field_from_x;
            let partition_from = match &computed_params.map_index_to_partition_data {
                Some(map_index_to_partition_data) => {
                    map_index_to_partition_data.get(&index_in_field_from)
                }
                _ => None,
            };

            let neuron_from_index = get_neuron_index(
                layer_params,
                computed_params,
                layer_from_x,
                layer_from_y,
                neuron_in_field_from_x,
                neuron_in_field_from_y,
            );

            for neuron_in_field_to_y in 0..layer_params.field_height {
                for neuron_in_field_to_x in 0..layer_params.field_width {
                    let index_in_field_to =
                        layer_params.field_width * neuron_in_field_to_y + neuron_in_field_to_x;

                    let partition_to = match &computed_params.map_index_to_partition_data {
                        Some(map_index_to_partition_data) => {
                            map_index_to_partition_data.get(&index_in_field_to)
                        }
                        _ => None,
                    };

                    let neuron_to_index = get_neuron_index(
                        layer_params,
                        computed_params,
                        layer_to_x,
                        layer_to_y,
                        neuron_in_field_to_x,
                        neuron_in_field_to_y,
                    );

                    let has_connections_by_partitions =
                        check_has_connections_by_partitions(partition_from, partition_to);

                    if has_connections_by_partitions {
                        has_connections
                            [neuron_to_index * computed_params.field_size + neuron_from_index] =
                            true;
                    }
                }
            }
        }
    }
}

fn fill_has_conntections(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
    offsets_2_to_1: &mut Vec<u64>,
    restore_offsets_1_to_2: &mut Vec<u64>,
) {
    let (last_layer_x, last_layer_y) = get_last_field(layer_params);

    let mut cur_layer_x = 0usize;
    let mut cur_layer_y = 0usize;

    let restoring_state_field_interval = synapse_params.signal_shift_interval as usize + 1;

    let mut index = 0usize;

    offsets_2_to_1[0] = computed_params.layer_size as u64;

    loop {
        let current_neuron_offset = get_neuron_index(
            layer_params,
            computed_params,
            cur_layer_x,
            cur_layer_y,
            0,
            0,
        ) as u64;

        if index > 0 && index % restoring_state_field_interval == 0 {
            restore_offsets_1_to_2.push(current_neuron_offset);
        }

        if cur_layer_x == last_layer_x && cur_layer_y == last_layer_y {
            return;
        }

        let (next_field_x, next_field_y) = get_next_field(layer_params, cur_layer_x, cur_layer_y);

        let next_neuron_offset = get_neuron_index(
            layer_params,
            computed_params,
            next_field_x,
            next_field_y,
            0,
            0,
        );

        offsets_2_to_1[next_neuron_offset / computed_params.field_size] = current_neuron_offset;

        cur_layer_x = next_field_x;
        cur_layer_y = next_field_y;

        index += 1;
    }
}

pub fn set_initial_connections(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
    mask: &SynapseMask,
) -> InitialConnections {
    let layer_size = layer_params.field_width
        * layer_params.field_height
        * layer_params.layer_width
        * layer_params.layer_height;

    let mut forward_synapses_1_to_2 = vec![0f32; layer_size * computed_params.field_size];
    let mut forward_synapses_2_to_1 = vec![0f32; layer_size * computed_params.field_size];

    let mut offsets_1_to_2 = vec![0u64; computed_params.field_count];

    for neuron_index in 0..layer_size {
        let neuron_in_field_index = neuron_index % computed_params.field_size;

        let synapse_index = neuron_in_field_index * layer_size + neuron_index;

        forward_synapses_1_to_2[synapse_index] = synapse_params.max_g;
        forward_synapses_2_to_1[synapse_index] = synapse_params.max_g;

        if neuron_in_field_index == 0 {
            offsets_1_to_2[neuron_index / computed_params.field_size] =
                (neuron_index - neuron_in_field_index) as u64;
        }
    }

    for offset_index in 0..computed_params.field_count {
        offsets_1_to_2.push((offset_index * computed_params.field_size) as u64);
    }

    let mut offsets_2_to_1 = vec![0u64; computed_params.field_count];
    let mut restore_offsets_1_to_2: Vec<u64> = vec![];

    let mut has_connections_by_partitions =
        vec![false; computed_params.field_size * computed_params.field_size];

    fill_conntected_fields(
        layer_params,
        computed_params,
        &mut has_connections_by_partitions,
        0,
        0,
        0,
        0,
    );

    let mut forward_distance_weights =
        vec![0.0_f32; computed_params.field_size * computed_params.field_size];

    fill_has_conntections(
        layer_params,
        computed_params,
        synapse_params,
        &mut offsets_2_to_1,
        &mut restore_offsets_1_to_2,
    );

    let restore_synapses_1_to_2 =
        vec![
            0.0_f32;
            restore_offsets_1_to_2.len() * computed_params.field_size * computed_params.field_size
        ];

    let mut restore_distance_weights_1_to_2: Vec<f32> =
        vec![
            0.0_f32;
            restore_offsets_1_to_2.len() * computed_params.field_size * computed_params.field_size
        ];

    for neuron_in_field_y in 0..layer_params.field_height {
        for neuron_in_field_x in 0..layer_params.field_width {
            apply_mask_to_block(
                layer_params,
                computed_params,
                &mut forward_distance_weights,
                mask,
                0,
                neuron_in_field_x,
                neuron_in_field_y,
                &has_connections_by_partitions,
            );

            for (index, offset) in restore_offsets_1_to_2.iter().enumerate() {
                let layer_field_index = (*offset as usize) / computed_params.field_size;

                let layer_x = layer_field_index % layer_params.layer_width;
                let layer_y = (layer_field_index - layer_x) / layer_params.layer_height;

                apply_mask_to_block(
                    layer_params,
                    computed_params,
                    &mut restore_distance_weights_1_to_2,
                    mask,
                    index,
                    layer_x * layer_params.field_width + neuron_in_field_x,
                    layer_y * layer_params.field_height + neuron_in_field_y,
                    &has_connections_by_partitions,
                );
            }
        }
    }

    InitialConnections {
        forward_synapses_1_to_2,
        forward_synapses_2_to_1,
        offsets_1_to_2,
        offsets_2_to_1,
        restore_offsets_1_to_2,
        restore_synapses_1_to_2,
        forward_distance_weights,
        restore_distance_weights_1_to_2,
    }
}

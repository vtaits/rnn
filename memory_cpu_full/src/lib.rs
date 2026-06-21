use rnn_architecture::Memory;

pub struct MemoryCpuFull {
    synapse_weights_1_to_2: Vec<f32>,
    synapse_weights_2_to_1: Vec<f32>,
    distances_1_to_2: Vec<f32>,
    distances_2_to_1: Vec<f32>,
    neurons_1: Vec<bool>,
    neurons_2: Vec<bool>,
    refract_intervals_1: Vec<u8>,
    refract_intervals_2: Vec<u8>,
    layer_size: usize,
}

impl MemoryCpuFull {
    pub fn new(
        layer_width: usize,
        layer_height: usize,
        field_width: usize,
        field_height: usize,
    ) -> Self {
        let layer_size = layer_width * layer_height * field_width * field_height;
        let synapse_count = layer_size * layer_size;

        Self {
            synapse_weights_1_to_2: vec![0.0; synapse_count],
            synapse_weights_2_to_1: vec![0.0; synapse_count],
            distances_1_to_2: vec![0.0; synapse_count],
            distances_2_to_1: vec![0.0; synapse_count],
            neurons_1: vec![false; layer_size],
            neurons_2: vec![false; layer_size],
            refract_intervals_1: vec![0; layer_size],
            refract_intervals_2: vec![0; layer_size],
            layer_size,
        }
    }
}

impl Memory for MemoryCpuFull {
    fn set_distance_1_to_2(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        distance: f32,
    ) {
        self.distances_1_to_2[neuron_from_index * self.layer_size + neuron_to_index] = distance;
    }

    fn set_distance_2_to_1(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        distance: f32,
    ) {
        self.distances_2_to_1[neuron_from_index * self.layer_size + neuron_to_index] = distance;
    }

    fn set_synapse_weight_1_to_2(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        weight: f32,
    ) {
        self.synapse_weights_1_to_2[neuron_from_index * self.layer_size + neuron_to_index] = weight;
    }

    fn set_synapse_weight_2_to_1(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        weight: f32,
    ) {
        self.synapse_weights_2_to_1[neuron_from_index * self.layer_size + neuron_to_index] = weight;
    }
}

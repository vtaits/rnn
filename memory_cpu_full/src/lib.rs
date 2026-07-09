use processor_cpu_full::ProcessorCpuFullSpecificMemory;
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
    field_size: usize,
}

impl MemoryCpuFull {
    pub fn new(
        field_width: usize,
        field_height: usize,
        layer_width: usize,
        layer_height: usize,
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
            field_size: field_width * field_height,
        }
    }
}

impl Memory for MemoryCpuFull {
    fn reset_neurons(&mut self) {
        self.neurons_1.fill(false);
        self.neurons_2.fill(false);
        self.refract_intervals_1.fill(0);
        self.refract_intervals_2.fill(0);
    }

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

impl ProcessorCpuFullSpecificMemory for MemoryCpuFull {
    fn get_layer_size(&self) -> usize {
        self.layer_size
    }

    fn read_output_field(&self) -> Vec<bool> {
        self.neurons_2[0..self.field_size].to_vec()
    }

    fn set_neuron_1(&mut self, index: usize, value: bool) {
        self.neurons_1[index] = value;
    }

    fn get_refract_interval_1(&self, index: usize) -> u8 {
        self.refract_intervals_1[index]
    }

    fn get_transfer_fields_1_to_2(
        &mut self,
    ) -> (&[bool], &mut [bool], &mut [u8], &[u8], &mut [f32], &[f32]) {
        (
            &self.neurons_1,
            &mut self.neurons_2,
            &mut self.refract_intervals_1,
            &self.refract_intervals_2,
            &mut self.synapse_weights_1_to_2,
            &self.distances_1_to_2,
        )
    }

    fn get_transfer_fields_2_to_1(
        &mut self,
    ) -> (&[bool], &mut [bool], &mut [u8], &[u8], &mut [f32], &[f32]) {
        (
            &self.neurons_2,
            &mut self.neurons_1,
            &mut self.refract_intervals_2,
            &self.refract_intervals_1,
            &mut self.synapse_weights_2_to_1,
            &self.distances_2_to_1,
        )
    }
}

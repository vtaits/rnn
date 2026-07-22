use ocl::{Buffer, ProQue};
use processor_opencl_full::ProcessorOpenCLFullSpecificMemory;
use rnn_architecture::Memory;

pub struct MemoryOpenCLFull {
    synapse_weights_1_to_2: Vec<f32>,
    synapse_weights_1_to_2_buffer: Option<Buffer<f32>>,
    synapse_weights_2_to_1: Vec<f32>,
    synapse_weights_2_to_1_buffer: Option<Buffer<f32>>,
    distances_1_to_2: Vec<f32>,
    distances_1_to_2_buffer: Option<Buffer<f32>>,
    distances_2_to_1: Vec<f32>,
    distances_2_to_1_buffer: Option<Buffer<f32>>,
    neurons_1: Vec<u8>,
    neurons_2: Vec<u8>,
    refract_intervals_1: Vec<u8>,
    refract_intervals_2: Vec<u8>,
    layer_size: usize,
    field_size: usize,
}

impl MemoryOpenCLFull {
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
            synapse_weights_1_to_2_buffer: None,
            synapse_weights_2_to_1: vec![0.0; synapse_count],
            synapse_weights_2_to_1_buffer: None,
            distances_1_to_2: vec![0.0; synapse_count],
            distances_1_to_2_buffer: None,
            distances_2_to_1: vec![0.0; synapse_count],
            distances_2_to_1_buffer: None,
            neurons_1: vec![0; layer_size],
            neurons_2: vec![0; layer_size],
            refract_intervals_1: vec![0; layer_size],
            refract_intervals_2: vec![0; layer_size],
            layer_size,
            field_size: field_width * field_height,
        }
    }
}

impl Memory for MemoryOpenCLFull {
    fn reset_neurons(&mut self) {
        self.neurons_1.fill(0);
        self.neurons_2.fill(0);
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

impl ProcessorOpenCLFullSpecificMemory for MemoryOpenCLFull {
    fn make_buffers(&mut self, pro_que: &ProQue) {
        self.synapse_weights_1_to_2_buffer = Some(
            Buffer::<f32>::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::flags::MEM_READ_WRITE)
                .len(self.synapse_weights_1_to_2.len())
                .copy_host_slice(&self.synapse_weights_1_to_2)
                .build()
                .unwrap(),
        );

        self.synapse_weights_2_to_1_buffer = Some(
            Buffer::<f32>::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::flags::MEM_READ_WRITE)
                .len(self.synapse_weights_2_to_1.len())
                .copy_host_slice(&self.synapse_weights_2_to_1)
                .build()
                .unwrap(),
        );

        self.distances_1_to_2_buffer = Some(
            Buffer::<f32>::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::flags::MEM_READ_ONLY)
                .len(self.distances_1_to_2.len())
                .copy_host_slice(&self.distances_1_to_2)
                .build()
                .unwrap(),
        );

        self.distances_2_to_1_buffer = Some(
            Buffer::<f32>::builder()
                .queue(pro_que.queue().clone())
                .flags(ocl::flags::MEM_READ_ONLY)
                .len(self.distances_2_to_1.len())
                .copy_host_slice(&self.distances_2_to_1)
                .build()
                .unwrap(),
        );
    }

    fn get_layer_size(&self) -> usize {
        self.layer_size
    }

    fn read_output_field(&self) -> Vec<bool> {
        self.neurons_2[0..self.field_size]
            .iter()
            .map(|value| if *value > 0 { true } else { false })
            .collect()
    }

    fn set_neuron_1(&mut self, index: usize, value: bool) {
        self.neurons_1[index] = if value { 1 } else { 0 };
    }

    fn get_refract_interval_1(&self, index: usize) -> u8 {
        self.refract_intervals_1[index]
    }

    fn get_transfer_fields_1_to_2(
        &mut self,
    ) -> (
        &[u8],
        &mut [u8],
        &mut [u8],
        &[u8],
        &Buffer<f32>,
        &Buffer<f32>,
    ) {
        (
            &self.neurons_1,
            &mut self.neurons_2,
            &mut self.refract_intervals_1,
            &self.refract_intervals_2,
            &self.synapse_weights_1_to_2_buffer.as_ref().unwrap(),
            &self.distances_1_to_2_buffer.as_ref().unwrap(),
        )
    }

    fn get_transfer_fields_2_to_1(
        &mut self,
    ) -> (
        &[u8],
        &mut [u8],
        &mut [u8],
        &[u8],
        &Buffer<f32>,
        &Buffer<f32>,
    ) {
        (
            &self.neurons_2,
            &mut self.neurons_1,
            &mut self.refract_intervals_2,
            &self.refract_intervals_1,
            &self.synapse_weights_2_to_1_buffer.as_ref().unwrap(),
            &self.distances_2_to_1_buffer.as_ref().unwrap(),
        )
    }
}

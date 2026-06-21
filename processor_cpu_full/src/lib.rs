use rnn_architecture::Processor;

pub trait ProcessorCpuFullMemory {
    fn get_layer_size(&self) -> usize;

    fn set_neuron_1(&mut self, index: usize, value: bool);

    fn set_neuron_2(&mut self, index: usize, value: bool);

    fn get_neurons_1(&self) -> &[bool];

    fn get_neurons_2(&self) -> &[bool];

    fn set_refract_interval_1(&mut self, index: usize, value: u8);

    fn set_refract_interval_2(&mut self, index: usize, value: u8);

    fn get_refract_interval_1(&self, index: usize) -> u8;

    fn get_refract_interval_2(&self, index: usize) -> u8;

    fn get_distance_1_to_2(&self, neuron_from_index: usize, neuron_to_index: usize) -> f32;

    fn get_distance_2_to_1(&self, neuron_from_index: usize, neuron_to_index: usize) -> f32;

    fn get_synapse_weight_1_to_2(&self, neuron_from_index: usize, neuron_to_index: usize) -> f32;

    fn get_synapse_weight_2_to_1(&self, neuron_from_index: usize, neuron_to_index: usize) -> f32;
}

pub struct ProcessorCpuFullParams {
    gamma_inc: f32,
    gamma_dec: f32,
    g_0: f32,
    g_dec: f32,
    g_inc: f32,
    min_g: f32,
    max_g: f32,
    field_width: usize,
    field_height: usize,
    threshold: f32,
}

pub struct ProcessorCpuFull {
    gamma_inc: f32,
    gamma_dec: f32,
    g_0: f32,
    g_dec: f32,
    g_inc: f32,
    min_g: f32,
    max_g: f32,
    field_size: usize,
    threshold: f32,
    memory: Box<dyn ProcessorCpuFullMemory>,
}

impl ProcessorCpuFull {
    pub fn new(params: ProcessorCpuFullParams, memory: Box<dyn ProcessorCpuFullMemory>) -> Self {
        let ProcessorCpuFullParams {
            gamma_inc,
            gamma_dec,
            g_0,
            g_dec,
            g_inc,
            min_g,
            max_g,
            field_height,
            field_width,
            threshold,
        } = params;

        Self {
            gamma_inc,
            gamma_dec,
            g_0,
            g_dec,
            g_inc,
            min_g,
            max_g,
            memory,
            threshold,
            field_size: field_width * field_height,
        }
    }
}

impl Processor for ProcessorCpuFull {
    fn input_signal(&mut self, signal: Vec<bool>) {
        let memory = self.memory.as_mut();

        for (index, value) in signal.iter().enumerate() {
            if index < self.field_size && memory.get_refract_interval_1(index) == 0 {
                memory.set_neuron_1(index, *value);
            }
        }
    }

    fn read_signal(&self) -> Vec<bool> {
        let memory = self.memory.as_ref();

        memory.get_neurons_1()[0..self.field_size].to_vec()
    }

    fn transfer_1_to_2(&mut self) {}

    fn transfer_2_to_1(&mut self) {}
}

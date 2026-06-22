use rnn_architecture::Processor;

use crate::{
    ProcessorCpuFullMemory, ProcessorCpuFullRefractRecounter, ProcessorCpuFullSignalTransferer,
};

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
    refract_recounter: Box<dyn ProcessorCpuFullRefractRecounter>,
    signal_transferer: Box<dyn ProcessorCpuFullSignalTransferer>,
    memory: Box<dyn ProcessorCpuFullMemory>,
}

impl ProcessorCpuFull {
    pub fn new(
        params: ProcessorCpuFullParams,
        memory: Box<dyn ProcessorCpuFullMemory>,
        signal_transferer: Box<dyn ProcessorCpuFullSignalTransferer>,
        refract_recounter: Box<dyn ProcessorCpuFullRefractRecounter>,
    ) -> Self {
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
            signal_transferer,
            refract_recounter,
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

    fn transfer_1_to_2(&mut self) {
        let memory = self.memory.as_mut();

        let (
            neurons_1,
            neurons_2,
            refract_intervals_1,
            refract_intervals_2,
            synapse_weights_1_to_2,
            distances_1_to_2,
        ) = memory.get_transfer_fields_1_to_2();

        self.signal_transferer.transfer(
            0.0,
            neurons_1,
            neurons_2,
            refract_intervals_2,
            synapse_weights_1_to_2,
            distances_1_to_2,
        );

        self.refract_recounter
            .recount(neurons_1, refract_intervals_1);
    }

    fn transfer_2_to_1(&mut self) {
        let memory = self.memory.as_mut();

        let (
            neurons_2,
            neurons_1,
            refract_intervals_2,
            refract_intervals_1,
            synapse_weights_2_to_1,
            distances_2_to_1,
        ) = memory.get_transfer_fields_2_to_1();

        self.signal_transferer.transfer(
            self.g_0,
            neurons_2,
            neurons_1,
            refract_intervals_1,
            synapse_weights_2_to_1,
            distances_2_to_1,
        );

        self.refract_recounter
            .recount(neurons_2, refract_intervals_2);
    }
}

use cpu_refract_recounter::CpuRefractRecounter;
use rnn_architecture::Processor;

use crate::{
    ProcessorCpuFullMemory, ProcessorCpuFullSignalTransferer,
};

pub struct ProcessorCpuFullParams {
    pub g_0: f32,
    pub field_width: usize,
    pub field_height: usize,
}

pub struct ProcessorCpuFull {
    g_0: f32,
    field_size: usize,
    refract_recounter: Box<dyn CpuRefractRecounter>,
    signal_transferer: Box<dyn ProcessorCpuFullSignalTransferer>,
    memory: Box<dyn ProcessorCpuFullMemory>,
}

impl ProcessorCpuFull {
    pub fn new(
        params: ProcessorCpuFullParams,
        memory: Box<dyn ProcessorCpuFullMemory>,
        signal_transferer: Box<dyn ProcessorCpuFullSignalTransferer>,
        refract_recounter: Box<dyn CpuRefractRecounter>,
    ) -> Self {
        let ProcessorCpuFullParams {
            g_0,
            field_height,
            field_width,
        } = params;

        Self {
            g_0,
            memory,
            signal_transferer,
            refract_recounter,
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

    fn reset_neurons(&mut self) {
        self.memory.reset_neurons();
    }

    fn read_signal(&self) -> Vec<bool> {
        let memory = self.memory.as_ref();

        memory.read_output_field()
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
            .recount(Box::new(neurons_1.iter().map(|value| *value)), refract_intervals_1);
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
            .recount(Box::new(neurons_2.iter().map(|value| *value)), refract_intervals_2);
    }
}

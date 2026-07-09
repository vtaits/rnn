use rnn_architecture::Memory;

pub trait ProcessorCpuFullSpecificMemory {
    fn get_layer_size(&self) -> usize;

    fn set_neuron_1(&mut self, index: usize, value: bool);

    fn read_output_field(&self) -> Vec<bool>;

    fn get_refract_interval_1(&self, index: usize) -> u8;

    fn get_transfer_fields_1_to_2(
        &mut self,
    ) -> (&[bool], &mut [bool], &mut [u8], &[u8], &mut [f32], &[f32]);

    fn get_transfer_fields_2_to_1(
        &mut self,
    ) -> (&[bool], &mut [bool], &mut [u8], &[u8], &mut [f32], &[f32]);
}

pub trait ProcessorCpuFullMemory: ProcessorCpuFullSpecificMemory + Memory {}

impl<T: ProcessorCpuFullSpecificMemory + Memory> ProcessorCpuFullMemory for T {}

use ocl::{Buffer, ProQue};
use rnn_architecture::Memory;

pub trait ProcessorOpenCLFullSpecificMemory {
    fn make_buffers(&mut self, pro_que: &ProQue);

    fn get_layer_size(&self) -> usize;

    fn set_neuron_1(&mut self, index: usize, value: bool);

    fn read_output_field(&self) -> Vec<bool>;

    fn get_refract_interval_1(&self, index: usize) -> u8;

    fn get_transfer_fields_1_to_2(
        &mut self,
    ) -> (
        &[u8],
        &mut [u8],
        &mut [u8],
        &[u8],
        &Buffer<f32>,
        &Buffer<f32>,
    );

    fn get_transfer_fields_2_to_1(
        &mut self,
    ) -> (
        &[u8],
        &mut [u8],
        &mut [u8],
        &[u8],
        &Buffer<f32>,
        &Buffer<f32>,
    );
}

pub trait ProcessorOpenCLFullMemory: ProcessorOpenCLFullSpecificMemory + Memory {}

impl<T: ProcessorOpenCLFullSpecificMemory + Memory> ProcessorOpenCLFullMemory for T {}

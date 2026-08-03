use ocl::{Buffer, Queue};

pub trait ProcessorOpenCLFullSignalTransferer {
    fn transfer(
        &self,
        g_0: f32,
        neurons_from: &[u8],
        neurons_to: &mut [u8],
        refract_intervals_to: &[u8],
        synapses: &Buffer<f32>,
        distances: &Buffer<f32>,
    );

    fn get_queue(&self) -> &Queue;
}

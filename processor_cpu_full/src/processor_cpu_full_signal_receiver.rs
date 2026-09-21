pub trait ProcessorCpuFullSignalReceiver {
    fn transfer(
        &self,
        neurons_1: &mut [bool],
        refract_intervals_1: &[u8],
        neurons_2: &[bool],
        synapses_1_to_2: &mut [f32],
    );
}

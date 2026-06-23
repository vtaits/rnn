use processor_cpu_full::ProcessorCpuFullSignalTransferer;

pub struct SignalTransfererCpuFull {}

impl SignalTransfererCpuFull {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessorCpuFullSignalTransferer for SignalTransfererCpuFull {
    fn transfer(
        &self,
        g_0: f32,
        neurons_from: &[bool],
        neurons_to: &mut [bool],
        refract_intervals_to: &[u8],
        synapses: &mut [f32],
        distances: &[f32],
    ) {
    }
}

use processor_cpu_full::ProcessorCpuFullSignalTransferer;

use crate::get_weight_coefficient::get_weight_coefficient;

pub struct SignalTransfererCpuFullParams {
    field_width: usize,
    field_height: usize,
    layer_width: usize,
    layer_height: usize,
    threshold: f32,
    gamma_inc: f32,
    gamma_dec: f32,
    g_inc: f32,
    g_dec: f32,
    min_g: f32,
    max_g: f32,
}

pub struct SignalTransfererCpuFull {
    layer_size: usize,
    threshold: f32,
    gamma_inc: f32,
    gamma_dec: f32,
    g_inc: f32,
    g_dec: f32,
    min_g: f32,
    max_g: f32,
}

impl SignalTransfererCpuFull {
    pub fn new(params: SignalTransfererCpuFullParams) -> Self {
        let SignalTransfererCpuFullParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
            threshold,
            gamma_inc,
            gamma_dec,
            g_inc,
            g_dec,
            min_g,
            max_g,
        } = params;

        Self {
            layer_size: field_width * field_height * layer_width * layer_height,
            threshold,
            gamma_inc,
            gamma_dec,
            g_inc,
            g_dec,
            min_g,
            max_g,
        }
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
        for neuron_to_index in 0..self.layer_size {
            if refract_intervals_to[neuron_to_index] > 0 {
                neurons_to[neuron_to_index] = false;
                continue;
            }

            let mut signal_sum: f32 = 0.0;

            for neuron_from_index in 0..self.layer_size {
                let synapse_index = neuron_from_index * self.layer_size + neuron_to_index;

                if neurons_from[neuron_from_index] {
                    let weight_to = synapses[synapse_index];

                    if weight_to > 0.0001 || weight_to < -0.0001 {
                        signal_sum += get_weight_coefficient(
                            &self.gamma_inc,
                            &self.gamma_dec,
                            &weight_to,
                            &g_0,
                        ) * distances[synapse_index];
                    }
                }
            }

            neurons_to[neuron_to_index] = signal_sum > self.threshold;
        }

        for neuron_from_index in 0..self.layer_size {
            if neurons_from[neuron_from_index] {
                continue;
            }

            for neuron_to_index in 0..self.layer_size {
                let synapse_index = neuron_from_index * self.layer_size + neuron_to_index;
                let prev_value = synapses[synapse_index];

                if refract_intervals_to[neuron_to_index] > 0 {
                    if synapses[synapse_index] > self.min_g {
                        synapses[synapse_index] = (prev_value - self.g_dec).max(self.min_g);
                    }
                } else if neurons_to[neuron_to_index] {
                    if synapses[synapse_index] < self.max_g {
                        synapses[synapse_index] = (prev_value + self.g_inc).min(self.max_g);
                    }
                }
            }
        }
    }
}

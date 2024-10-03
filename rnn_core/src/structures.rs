use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ocl::{Kernel, ProQue};
use serde_derive::{Deserialize, Serialize};

pub struct CompiledKernel {
    pub kernel: Arc<Mutex<Kernel>>,
    pub pro_que: ProQue,
}

#[derive(Serialize, Deserialize)]
pub struct SynapseParams {
    pub alpha: f32,
    pub gamma: f32,
    pub g_dec: f32,
    pub g_inc: f32,
    pub g_0: f32,
    pub max_g: f32,
    pub initial_strong_g: f32,
    pub h: u8,
    pub threshold: f32,
    pub refract_interval: u8,
    pub signal_shift_interval: u8,
}

#[derive(Serialize, Deserialize)]
pub struct LayerParams {
    /// Width in neurons of one field
    pub field_width: usize,
    /// Height in neurons of one field
    pub field_height: usize,
    /// Width in fields of one layer
    pub layer_width: usize,
    // Height in fields of one layer
    pub layer_height: usize,
}

pub struct SynapseMask {
    pub size: usize,
    pub offset: usize,
    pub mask: Array2<f32>,
}

#[derive(Serialize)]
pub struct NetworkDumpSerialize<'a> {
    pub accumulated_weights_1_to_2: &'a Array2<f32>,
    // acumulated weights of synapses from the second layer to the first layer
    pub accumulated_weights_2_to_1: &'a Array2<f32>,
    // distance weights of synapses from the first layer to the second layer
    pub distance_weights_1_to_2: &'a Array2<f32>,
    // distance weights of synapses from the second layer to the first layer
    pub distance_weights_2_to_1: &'a Array2<f32>,
    pub neurons_1: &'a Array1<f32>,
    // neuron states at the second layer
    pub neurons_2: &'a Array1<f32>,
    // timeouts of neuron refract states of the first layer
    pub refract_intervals_1: &'a Array1<u8>,
    // timeouts of neuron refract states of the second layer
    pub refract_intervals_2: &'a Array1<u8>,
    pub layer_params: &'a LayerParams,
    pub synapse_params: &'a SynapseParams,
}

#[derive(Deserialize)]
pub struct NetworkDumpDeserialize {
    pub accumulated_weights_1_to_2: Array2<f32>,
    // acumulated weights of synapses from the second layer to the first layer
    pub accumulated_weights_2_to_1: Array2<f32>,
    // distance weights of synapses from the first layer to the second layer
    pub distance_weights_1_to_2: Array2<f32>,
    // distance weights of synapses from the second layer to the first layer
    pub distance_weights_2_to_1: Array2<f32>,
    pub neurons_1: Array1<f32>,
    // neuron states at the second layer
    pub neurons_2: Array1<f32>,
    // timeouts of neuron refract states of the first layer
    pub refract_intervals_1: Array1<u8>,
    // timeouts of neuron refract states of the second layer
    pub refract_intervals_2: Array1<u8>,
    pub layer_params: LayerParams,
    pub synapse_params: SynapseParams,
}

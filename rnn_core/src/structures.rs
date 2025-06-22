use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ocl::{Kernel, ProQue};
use serde_derive::{Deserialize, Serialize};

pub struct ComputedParams {
    // number of fields in one layer
    pub field_count: usize,
    // number of neurons in one field
    pub field_size: usize,
    // number of neurons in one row of fields
    pub row_size: usize,
    // number of neurons in one row of neurons
    pub row_width: usize,
    // number of neurons in one column of neurons
    pub column_height: usize,
    // number of empty shifts that should be fulfiled after the last step of the prediction
    pub prediction_rest_shifts: usize,
    // maximal number of excited neurons
    pub excited_neurons_limit: usize,
}

pub struct CompiledKernel {
    pub kernel: Arc<Mutex<Kernel>>,
    pub pro_que: ProQue,
}

#[derive(Serialize, Deserialize)]
pub struct SynapseParams {
    pub alpha: f32,
    pub gamma_inc: f32,
    pub gamma_dec: f32,
    pub g_dec: f32,
    pub g_inc: f32,
    pub g_0: f32,
    pub min_g: f32,
    pub max_g: f32,
    pub initial_strong_g: f32,
    pub h: f32,
    pub threshold: f32,
    pub refract_interval: u8,
    pub signal_shift_interval: u8,
    pub signal_rest_shift_limit: Option<u8>,
    /// Apply additional signals that copy original signal with the shift
    pub signal_copy_shifts: Option<Vec<(i8, i8)>>,
    /// Percentage of maximum number of excited neurons after the step of neural network
    pub excite_neuron_limit: f32,
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
    // neuron states at the first layer
    pub neurons_1: &'a Array1<u8>,
    // neuron states at the second layer
    pub neurons_2: &'a Array1<u8>,
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
    // neuron states at the first layer
    pub neurons_1: Array1<u8>,
    // neuron states at the second layer
    pub neurons_2: Array1<u8>,
    // timeouts of neuron refract states of the first layer
    pub refract_intervals_1: Array1<u8>,
    // timeouts of neuron refract states of the second layer
    pub refract_intervals_2: Array1<u8>,
    pub layer_params: LayerParams,
    pub synapse_params: SynapseParams,
}

pub enum Action {
    ApplyRest(Vec<bool>, u8),
    EmptyShift1to2,
    EmptyShift2to1,
    /**
     * 0 - signal
     * 1 - whether it a source signal that should be taken into account in prediction process
     */
    InputSignal(Vec<bool>, bool),
}

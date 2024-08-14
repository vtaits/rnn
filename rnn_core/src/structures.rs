use ndarray::Array2;
use ocl::{Kernel, ProQue};

pub struct CompiledKernel {
    pub kernel: Kernel,
    pub pro_que: ProQue,
}

pub struct DataAdapter<T> {
    pub binary_to_data: Box<dyn Fn(Vec<bool>) -> Result<T, ()>>,
    pub data_to_binary: Box<dyn Fn(T) -> Result<Vec<bool>, ()>>,
}

pub struct SynapseParams {
    pub alpha: f32,
    pub gamma: f32,
    pub g_dec: f32,
    pub g_inc: f32,
    pub g_0: f32,
    pub initial_strong_g: f32,
    pub h: u8,
    pub threshold: f32,
    pub refract_interval: u8,
}

pub struct NetworkParams {
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

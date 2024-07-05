use ndarray::Array2;

pub struct SynapseParams {
    pub alpha: f32,
    pub h: u8,
    pub threshold: f32,
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
    pub mask: Array2<f32>
}

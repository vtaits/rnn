#[derive(Clone, Debug)]
pub struct RedefineParams {
    pub alpha: f32,
    pub gamma_dec: f32,
    pub gamma_inc: f32,
    pub g_dec: f32,
    pub g_inc: f32,
    pub g_0: f32,
    pub h: f32,
    pub threshold: f32,
}

pub struct InitDataLayerParams {
    pub train: bool,
    pub start_index: Option<usize>,
    pub end_index: Option<usize>,
    pub start_measurement_index: Option<usize>,
    pub end_measurement_index: Option<usize>,
    pub redefine_params: Option<RedefineParams>,
}

use ndarray::{Array1, Array2};
pub struct NetworkParams {
  /// Width in neurons of one field
  field_width: usize,
  /// Height in neurons of one field
  field_height: usize,
  /// Width in fields of one layer
  layer_width: usize,
  // Height in fields of one layer
  layer_height: usize,
}

pub struct Network {
  // number of neurons in one layer
  layer_size: usize,
  // neuron states at the first layer
  neurons_1: Array1<u8>,
  // neuron states at the second layer
  neurons_2: Array1<u8>,
  // weights of synapses from the first layer to the second layer
  weights_1_to_2: Array2<f32>,
  // weights of synapses from the second layer to the first layer
  weights_2_to_1: Array2<f32>,
}

impl Network {
  pub fn new(params: &NetworkParams) -> Network {
    let layer_size = params.field_width * params.field_height * params.layer_width * params.layer_height;

    let neurons_1 = Array1::<u8>::zeros(layer_size);
    let neurons_2 = Array1::<u8>::zeros(layer_size);

    let weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    return Network {
      layer_size,
      neurons_1,
      neurons_2,
      weights_1_to_2,
      weights_2_to_1,
    };
  }
}
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

// TO DO: bit-vec / bitfield
pub struct Network {
  // number of neurons
  field_size: usize,
  // number of neurons in one layer
  layer_size: usize,
  // neuron states at the first layer
  neurons_1: Array1<f32>,
  // neuron states at the second layer
  neurons_2: Array1<f32>,
  // weights of synapses from the first layer to the second layer
  weights_1_to_2: Array2<f32>,
  // weights of synapses from the second layer to the first layer
  weights_2_to_1: Array2<f32>,
}

fn set_initial_connections(
  params: &NetworkParams,
) -> (
  // weights of synapses from the first layer to the second layer
  Array2<f32>,
  // weights of synapses from the second layer to the first layer
  Array2<f32>,
) {
  let layer_width = params.field_width * params.layer_width;
  let layer_size = params.field_width * params.field_height * params.layer_width * params.layer_height;

  let mut weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
  let mut weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

  for layer_x in 0..params.layer_width - 1 {
    for layer_y in 0..params.layer_height - 1 {
      let layer_2_to_1_x = if layer_y % 2 == 0 {
        if layer_x == params.layer_width - 1 { 0 } else { layer_x + 1 }
      } else {
        if layer_x == 0 { params.layer_width - 1 } else { layer_x - 1 }
      };

      let layer_2_to_1_y = if layer_x == params.layer_width - 1 {
        if layer_y == params.layer_height - 1 { 0 } else {layer_y + 1 }
      } else {
        layer_y
      };

      for neuron_in_field_x in 0..params.field_width - 1 {
        for neuron_in_field_y in 0..params.field_height - 1 {
          let neuron_x = params.field_width * layer_x + neuron_in_field_x;
          let neuron_y = params.field_height * layer_y + neuron_in_field_y;
          let neuron_index = neuron_y * layer_width + neuron_x;

          let neuron_2_to_1_x = params.field_width * layer_2_to_1_x + neuron_in_field_x;
          let neuron_2_to_1_y = params.field_height * layer_2_to_1_y + neuron_in_field_y;
          let neuron_2_to_1_index = neuron_2_to_1_y * neuron_2_to_1_x + neuron_x;

          weights_1_to_2[[neuron_index, neuron_index]] = 1.0;
          weights_2_to_1[[neuron_index, neuron_2_to_1_index]] = 1.0;
        }
      }
    }
  }

  return (
    weights_1_to_2,
    weights_2_to_1,
  );
}

impl Network {
  pub fn new(params: &NetworkParams) -> Network {
    let field_size = params.field_width * params.field_height;
    let layer_size = field_size * params.layer_width * params.layer_height;

    let neurons_1 = Array1::<f32>::zeros(layer_size);
    let neurons_2 = Array1::<f32>::zeros(layer_size);

    let (weights_1_to_2, weights_2_to_1) = set_initial_connections(&params);

    return Network {
      field_size,
      layer_size,
      neurons_1,
      neurons_2,
      weights_1_to_2,
      weights_2_to_1,
    };
  }

  pub fn tick(&mut self, data: &Vec<bool>) {
    if &data.len() > &self.field_size {
      panic!("The length of the data chunk should be less than or equal to the size of one field");
    }

    for (pos, value) in data.iter().enumerate() {
      self.neurons_1[[pos]] = if *value { 1.0 } else { 0.0 };
    }

    let neurons_2_next: Array1<f32>  = self.weights_1_to_2.dot(&self.neurons_1);
    let neurons_1_next: Array1<f32>  = self.weights_2_to_1.dot(&self.neurons_2);

    self.neurons_1 = neurons_1_next;
    self.neurons_2 = neurons_2_next;
  }
}
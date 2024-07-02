use ndarray::{Array1, Array2};
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

struct ComputedParams {
  // number of neurons
  field_size: usize,
  field_width: usize,
  // number of neurons in one row of fields
  row_size: usize,
}

// TO DO: bit-vec / bitfield
pub struct Network {
  computed_params: ComputedParams,
  field_width: usize,
  field_height: usize,
  layer_width: usize,
  layer_height: usize,
  // number of neurons
  field_size: usize,
  // number of neurons in one layer
  layer_size: usize,
  // neuron states at the first layer
  neurons_1: Array1<f32>,
  // neuron states at the second layer
  neurons_2: Array1<f32>,
  // number of neurons in one row of fields
  row_size: usize,
  // weights of synapses from the first layer to the second layer
  weights_1_to_2: Array2<f32>,
  // weights of synapses from the second layer to the first layer
  weights_2_to_1: Array2<f32>,
}

fn get_neuron_index(
  computed_params: &ComputedParams,
  layer_x: usize,
  layer_y: usize,
  neuron_in_field_x: usize,
  neuron_in_field_y: usize,
) -> usize {
  let layer_offset = computed_params.row_size * layer_y + computed_params.field_size * layer_x;
  let field_offset = computed_params.field_width * neuron_in_field_y + neuron_in_field_x;

  return layer_offset + field_offset;
}

fn get_next_field(params: &NetworkParams, field_x: usize, field_y: usize) -> (usize, usize) {
  let layer_2_to_1_x = if field_y % 2 == 0 {
    if field_x == params.layer_width - 1 { field_x } else { field_x + 1 }
  } else {
    if field_x == 0 { field_x } else { field_x - 1 }
  };

  let layer_2_to_1_y = if field_y % 2 == 0 {
    if field_x == params.layer_width - 1 {
      if field_y == params.layer_height - 1 { 0 } else { field_y + 1 }
    } else { field_y }
  } else {
    if field_x == 0 {
      if field_y == params.layer_height - 1 { 0 } else { field_y + 1 }
    } else { field_y }
  };

  return (layer_2_to_1_x, layer_2_to_1_y);
}

fn set_initial_connections(
  computed_params: &ComputedParams,
  params: &NetworkParams,
) -> (
  // weights of synapses from the first layer to the second layer
  Array2<f32>,
  // weights of synapses from the second layer to the first layer
  Array2<f32>,
) {
  let layer_size = params.field_width * params.field_height * params.layer_width * params.layer_height;

  let mut weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
  let mut weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

  for layer_y in 0..params.layer_height {
    for layer_x in 0..params.layer_width {
      let (layer_2_to_1_x, layer_2_to_1_y) = get_next_field(&params, layer_x, layer_y);

      println!("{} {} -> {}, {}", layer_x, layer_y, layer_2_to_1_x, layer_2_to_1_y);

      for neuron_in_field_y in 0..params.field_height {
        for neuron_in_field_x in 0..params.field_width {
          let neuron_index = get_neuron_index(computed_params, layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);
          let neuron_2_to_1_index = get_neuron_index(computed_params, layer_2_to_1_x, layer_2_to_1_y, neuron_in_field_x, neuron_in_field_y);

          weights_1_to_2[[neuron_index, neuron_index]] = 1.0;
          weights_2_to_1[[neuron_2_to_1_index, neuron_index]] = 1.0;
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
    let NetworkParams {
      field_width,
      field_height,
      layer_width,
      layer_height,
    } = params;

    let field_size = field_width * field_height;
    let row_size = field_size * layer_width;

    let layer_size = field_size * layer_width * layer_height;

    let neurons_1 = Array1::<f32>::zeros(layer_size);
    let neurons_2 = Array1::<f32>::zeros(layer_size);

    let computed_params = ComputedParams {
      field_size,
      field_width: *field_width,
      row_size,
    };

    let (weights_1_to_2, weights_2_to_1) = set_initial_connections(&computed_params, &params);

    return Network {
      computed_params,
      field_width: *field_width,
      field_height: *field_height,
      layer_width: *layer_width,
      layer_height: *layer_height,
      field_size,
      layer_size,
      neurons_1,
      neurons_2,
      row_size,
      weights_1_to_2,
      weights_2_to_1,
    };
  }

  pub fn tick(&mut self, data: &Vec<bool>) {
    let data_len = data.len();

    if data_len > self.field_size {
      panic!("The length of the data chunk should be less than or equal to the size of one field");
    }

    for (pos, value) in data.iter().enumerate() {
      self.neurons_1[[pos]] = if *value { 1.0 } else { 0.0 };
    }

    for pos in data_len..self.field_size {
      self.neurons_1[[pos]] = 0.0;
    }

    let neurons_2_next: Array1<f32>  = self.weights_1_to_2.dot(&self.neurons_1);
    let neurons_1_next: Array1<f32>  = self.weights_2_to_1.dot(&self.neurons_2);

    self.neurons_1 = neurons_1_next;
    self.neurons_2 = neurons_2_next;
  }

  pub fn print_states(&self) {
    print!("STATES:\n\n");
    print!("LAYER 1:\n");
    self.print_state(&self.neurons_1);
    print!("LAYER 2:\n");
    self.print_state(&self.neurons_2);
    print!("\n");
    print!("\n");
  }

  fn print_state(&self, layer: &Array1<f32>) {
    for layer_y in 0..self.layer_height {
      for neuron_in_field_y in 0..self.field_height {
        for layer_x in 0..self.layer_width {
          for neuron_in_field_x in 0..self.field_width {
            let neuron_index = get_neuron_index(&self.computed_params, layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);

            print!("{}", if layer[[neuron_index]] == 1.0 { "+" } else { "." });
          }

          print!(" ");
        }

        print!("\n");
      }

      print!("\n");
    }
  }
}
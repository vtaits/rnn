use ndarray::{Array1, Array2};

use crate::{get_synapse_mask::get_synapse_mask, spiral::get_next_field, structures::{NetworkParams, SynapseMask, SynapseParams}};

struct ComputedParams {
  // number of neurons in one field
  field_size: usize,
  field_width: usize,
  // number of neurons in one row of fields
  row_size: usize,
  // number of neurons in one row of neurons
  row_width: usize,
  // number of neurons in one column of neurons
  column_height: usize,
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

fn get_neuron_coordinates(
  params: &NetworkParams,
  layer_x: usize,
  layer_y: usize,
  neuron_in_field_x: usize,
  neuron_in_field_y: usize,
) -> (usize, usize) {
  let layer_x_offset = params.field_width * layer_x;
  let layer_y_offset = params.field_height * layer_y;

  let field_x_offset = neuron_in_field_x;
  let field_y_offset = neuron_in_field_y;

  return (layer_x_offset + field_x_offset, layer_y_offset + field_y_offset);
}

fn get_neuron_index_by_coordinates(
  params: &NetworkParams,
  computed_params: &ComputedParams,
  neuron_x: usize,
  neuron_y: usize,
) -> usize {
  let neuron_in_field_x = neuron_x % params.field_width;
  let neuron_in_field_y = neuron_y % params.field_height;
  
  let layer_x = (neuron_x - neuron_in_field_x) / params.field_width;
  let layer_y = (neuron_y - neuron_in_field_y) / params.field_height;

  return get_neuron_index(computed_params, layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);
}

fn apply_mask(
  params: &NetworkParams,
  computed_params: &ComputedParams,
  base_neuron_index: usize,
  weights: &mut Array2<f32>,
  mask: &SynapseMask,
  x: usize,
  y: usize,
) {
  for iter_x in 0..mask.size {
    let offset_x = iter_x as i32 - mask.offset as i32;
    let neuron_x = (x as i32) + offset_x;

    if neuron_x < 0 || (neuron_x as usize) > computed_params.row_width - 1 {
      continue;
    }
    
    for iter_y in 0..mask.size {
      let offset_y = iter_y as i32 - mask.offset as i32;
      let neuron_y = (y as i32) + offset_y;
  
      if neuron_y < 0 || (neuron_y as usize) > computed_params.column_height - 1 {
        continue;
      }

      let target_neuron_index = get_neuron_index_by_coordinates(params, computed_params, neuron_x as usize, neuron_y as usize);

      let value = mask.mask[[iter_x, iter_y]];

      weights[[target_neuron_index, base_neuron_index]] = value;
    }
  }
}

fn set_initial_connections(
  computed_params: &ComputedParams,
  params: &NetworkParams,
  mask: &SynapseMask,
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

      for neuron_in_field_y in 0..params.field_height {
        for neuron_in_field_x in 0..params.field_width {
          let neuron_index = get_neuron_index(computed_params, layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);

          let (x, y) = get_neuron_coordinates(params, layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);

          apply_mask(params, computed_params, neuron_index, &mut weights_1_to_2, mask, x, y);

          let (x_2_to_1, y_2_to_1) = get_neuron_coordinates(params, layer_2_to_1_x, layer_2_to_1_y, neuron_in_field_x, neuron_in_field_y);

          apply_mask(params, computed_params, neuron_index, &mut weights_2_to_1, mask, x_2_to_1, y_2_to_1);
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
  pub fn new(
    params: &NetworkParams,
    synapse_params: &SynapseParams,
  ) -> Network {
    let NetworkParams {
      field_width,
      field_height,
      layer_width,
      layer_height,
    } = params;

    let field_size = field_width * field_height;
    let row_size = field_size * layer_width;
    let row_width = field_width * layer_width;
    let column_height = field_height * layer_height;

    let layer_size = field_size * layer_width * layer_height;

    let neurons_1 = Array1::<f32>::zeros(layer_size);
    let neurons_2 = Array1::<f32>::zeros(layer_size);

    let computed_params = ComputedParams {
      field_size,
      field_width: *field_width,
      row_size,
      row_width,
      column_height,
    };

    let mask = get_synapse_mask(synapse_params);

    let (weights_1_to_2, weights_2_to_1) = set_initial_connections(&computed_params, &params, &mask);

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

            print!("{} ", layer[[neuron_index]]);
          }

          print!(" ");
        }

        print!("\n");
      }

      print!("\n");
    }
  }
}
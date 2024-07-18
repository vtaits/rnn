use ndarray::{Array1, Array2};

use crate::{
    apply_synapses::{apply_synapses, build_kernel, CompiledKernel},
    get_synapse_mask::get_synapse_mask,
    spiral::get_next_field,
    structures::{NetworkParams, SynapseMask, SynapseParams},
};

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
pub struct Network<'a> {
    computed_params: ComputedParams,
    // acumulated weights of synapses from the first layer to the second layer
    accumulated_weights_1_to_2: Array2<f32>,
    // acumulated weights of synapses from the second layer to the first layer
    accumulated_weights_2_to_1: Array2<f32>,
    // distance weights of synapses from the first layer to the second layer
    distance_weights_1_to_2: Array2<f32>,
    // distance weights of synapses from the second layer to the first layer
    distance_weights_2_to_1: Array2<f32>,
    field_width: usize,
    field_height: usize,
    // compiled kernel for opencl computations
    kernel: CompiledKernel,
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
    // timeouts of neuron refract states of the first layer
    refract_intervals_1: Array1<u8>,
    // timeouts of neuron refract states of the second layer
    refract_intervals_2: Array1<u8>,
    params: &'a NetworkParams,
    synapse_params: SynapseParams,
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

    return (
        layer_x_offset + field_x_offset,
        layer_y_offset + field_y_offset,
    );
}

fn get_neuron_full_coordinates(
    params: &NetworkParams,
    neuron_x: usize,
    neuron_y: usize,
) -> (usize, usize, usize, usize) {
    let neuron_in_field_x = neuron_x % params.field_width;
    let neuron_in_field_y = neuron_y % params.field_height;

    let layer_x = (neuron_x - neuron_in_field_x) / params.field_width;
    let layer_y = (neuron_y - neuron_in_field_y) / params.field_height;

    return (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y);
}

fn get_neuron_index_by_coordinates(
    params: &NetworkParams,
    computed_params: &ComputedParams,
    neuron_x: usize,
    neuron_y: usize,
) -> usize {
    let (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y) =
        get_neuron_full_coordinates(params, neuron_x, neuron_y);

    return get_neuron_index(
        computed_params,
        layer_x,
        layer_y,
        neuron_in_field_x,
        neuron_in_field_y,
    );
}

fn apply_mask(
    params: &NetworkParams,
    computed_params: &ComputedParams,
    base_neuron_index: usize,
    distance_weights: &mut Array2<f32>,
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

            let target_neuron_index = get_neuron_index_by_coordinates(
                params,
                computed_params,
                neuron_x as usize,
                neuron_y as usize,
            );

            let value = mask.mask[[iter_x, iter_y]];

            distance_weights[[target_neuron_index, base_neuron_index]] = value;
        }
    }
}

fn set_initial_connections(
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
    params: &NetworkParams,
    mask: &SynapseMask,
) -> (
    // distance_weights of synapses from the first layer to the second layer
    Array2<f32>,
    // distance_weights of synapses from the second layer to the first layer
    Array2<f32>,
    // accumulated of synapses from the first layer to the second layer
    Array2<f32>,
    // accumulated of synapses from the second layer to the first layer
    Array2<f32>,
) {
    let layer_size =
        params.field_width * params.field_height * params.layer_width * params.layer_height;

    let mut distance_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut distance_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let mut accumulated_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut accumulated_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    for layer_y in 0..params.layer_height {
        for layer_x in 0..params.layer_width {
            let (layer_2_to_1_x, layer_2_to_1_y) = get_next_field(&params, layer_x, layer_y);

            for neuron_in_field_y in 0..params.field_height {
                for neuron_in_field_x in 0..params.field_width {
                    let neuron_index = get_neuron_index(
                        computed_params,
                        layer_x,
                        layer_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );
                    let neuron_2_to_1_index = get_neuron_index(
                        computed_params,
                        layer_2_to_1_x,
                        layer_2_to_1_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    accumulated_weights_1_to_2[[neuron_index, neuron_index]] =
                        synapse_params.initial_strong_g;
                    accumulated_weights_2_to_1[[neuron_2_to_1_index, neuron_index]] =
                        synapse_params.initial_strong_g;

                    let (x, y) = get_neuron_coordinates(
                        params,
                        layer_x,
                        layer_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    apply_mask(
                        params,
                        computed_params,
                        neuron_index,
                        &mut distance_weights_1_to_2,
                        mask,
                        x,
                        y,
                    );

                    let (x_2_to_1, y_2_to_1) = get_neuron_coordinates(
                        params,
                        layer_2_to_1_x,
                        layer_2_to_1_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    apply_mask(
                        params,
                        computed_params,
                        neuron_index,
                        &mut distance_weights_2_to_1,
                        mask,
                        x_2_to_1,
                        y_2_to_1,
                    );
                }
            }
        }
    }

    return (
        distance_weights_1_to_2,
        distance_weights_2_to_1,
        accumulated_weights_1_to_2,
        accumulated_weights_2_to_1,
    );
}

impl<'a> Network<'a> {
    pub fn new(params: &NetworkParams, synapse_params: SynapseParams) -> Network {
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

        let computed_params = ComputedParams {
            field_size,
            field_width: *field_width,
            row_size,
            row_width,
            column_height,
        };

        let mask = get_synapse_mask(&synapse_params);

        let (
            distance_weights_1_to_2,
            distance_weights_2_to_1,
            accumulated_weights_1_to_2,
            accumulated_weights_2_to_1,
        ) = set_initial_connections(&computed_params, &synapse_params, &params, &mask);

        let kernel = build_kernel(layer_size).unwrap();

        return Network {
            accumulated_weights_1_to_2,
            accumulated_weights_2_to_1,
            computed_params,
            distance_weights_1_to_2,
            distance_weights_2_to_1,
            field_width: *field_width,
            field_height: *field_height,
            kernel,
            layer_width: *layer_width,
            layer_height: *layer_height,
            field_size,
            layer_size,
            neurons_1: Array1::<f32>::zeros(layer_size),
            neurons_2: Array1::<f32>::zeros(layer_size),
            refract_intervals_1: Array1::<u8>::zeros(layer_size),
            refract_intervals_2: Array1::<u8>::zeros(layer_size),
            params,
            synapse_params,
        };
    }

    pub fn get_params(&self) -> &NetworkParams {
        return self.params;
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

        let apply_synapses_result_2 = apply_synapses(
            &self.kernel,
            self.layer_size,
            &self.accumulated_weights_1_to_2,
            &self.distance_weights_1_to_2,
            &self.neurons_1,
            &self.refract_intervals_2,
            self.synapse_params.refract_interval,
            self.synapse_params.threshold,
            self.synapse_params.gamma,
            0.0,
        )
        .unwrap();

        let apply_synapses_result_1 = apply_synapses(
            &self.kernel,
            self.layer_size,
            &self.accumulated_weights_2_to_1,
            &self.distance_weights_2_to_1,
            &self.neurons_2,
            &self.refract_intervals_1,
            self.synapse_params.refract_interval,
            self.synapse_params.threshold,
            self.synapse_params.gamma,
            self.synapse_params.g_0,
        )
        .unwrap();

        self.neurons_1 = apply_synapses_result_1.next_neurons;
        self.refract_intervals_1 = apply_synapses_result_1.next_refract_intervals;
        self.neurons_2 = apply_synapses_result_2.next_neurons;
        self.refract_intervals_2 = apply_synapses_result_2.next_refract_intervals;
    }

    pub fn print_states(&self) {
        println!("STATES:");
        println!();
        println!("LAYER 1:");
        self.print_state(&self.neurons_1);
        println!("LAYER 2:");
        self.print_state(&self.neurons_2);
        println!();
        println!();
    }

    fn print_state(&self, layer: &Array1<f32>) {
        for layer_y in 0..self.layer_height {
            for neuron_in_field_y in 0..self.field_height {
                for layer_x in 0..self.layer_width {
                    for neuron_in_field_x in 0..self.field_width {
                        let neuron_index = get_neuron_index(
                            &self.computed_params,
                            layer_x,
                            layer_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        print!(
                            "{} ",
                            if layer[[neuron_index]] > 0.5 {
                                "+"
                            } else {
                                "."
                            }
                        );
                    }

                    print!(" ");
                }

                println!();
            }

            println!();
        }
    }

    pub fn get_layer_dimensions(&self) -> (usize, usize) {
        return (
            self.computed_params.row_width,
            self.computed_params.column_height,
        );
    }

    pub fn get_neuron_refract_timeout(
        &self,
        layer_index: u8,
        layer_x: usize,
        layer_y: usize,
        neuron_in_field_x: usize,
        neuron_in_field_y: usize,
    ) -> u8 {
        let refract_intervals = if layer_index == 1 {
            &self.refract_intervals_1
        } else {
            &self.refract_intervals_2
        };

        let neuron_index = get_neuron_index(
            &self.computed_params,
            layer_x,
            layer_y,
            neuron_in_field_x,
            neuron_in_field_y,
        );

        return refract_intervals[[neuron_index]];
    }

    pub fn get_neuron_full_coordinates(
        &self,
        neuron_x: usize,
        neuron_y: usize,
    ) -> (usize, usize, usize, usize) {
        return get_neuron_full_coordinates(&self.params, neuron_x, neuron_y);
    }

    fn get_neuron_weights(
      &self,
      weights_layer: &Array2<f32>,
      neuron_x: usize,
      neuron_y: usize,
  ) -> Array2<f32> {
      let mut res = Array2::<f32>::zeros([
          self.computed_params.row_width,
          self.computed_params.column_height,
      ]);

      let neuron_index =
          get_neuron_index_by_coordinates(self.params, &self.computed_params, neuron_x, neuron_y);

      for layer_y in 0..self.layer_height {
          for neuron_in_field_y in 0..self.field_height {
              for layer_x in 0..self.layer_width {
                  for neuron_in_field_x in 0..self.field_width {
                      let target_neuron_index = get_neuron_index(
                          &self.computed_params,
                          layer_x,
                          layer_y,
                          neuron_in_field_x,
                          neuron_in_field_y,
                      );

                      let (target_x, target_y) = get_neuron_coordinates(
                          self.params,
                          layer_x,
                          layer_y,
                          neuron_in_field_x,
                          neuron_in_field_y,
                      );

                      res[[target_x, target_y]] =
                          weights_layer[[target_neuron_index, neuron_index]];
                  }
              }
          }
      }

      return res;
  }

    pub fn get_neuron_accumulated_weights(
        &self,
        layer_index: u8,
        neuron_x: usize,
        neuron_y: usize,
    ) -> Array2<f32> {
        let weights_layer = if layer_index == 1 {
            &self.accumulated_weights_1_to_2
        } else {
            &self.accumulated_weights_2_to_1
        };

        return self.get_neuron_weights(weights_layer, neuron_x, neuron_y);
    }

    pub fn get_neuron_distance_weights(
      &self,
      layer_index: u8,
      neuron_x: usize,
      neuron_y: usize,
  ) -> Array2<f32> {
      let weights_layer = if layer_index == 1 {
          &self.distance_weights_1_to_2
      } else {
          &self.distance_weights_2_to_1
      };

      return self.get_neuron_weights(weights_layer, neuron_x, neuron_y);
  }
}

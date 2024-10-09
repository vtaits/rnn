use std::io::prelude::*;
use std::io::Write;

use flate2::Compression;
use flate2::{read::GzDecoder, write::GzEncoder};
use ndarray::{Array1, Array2};

use crate::logger::Logger;
use crate::recount_refract_intervals::recount_refract_intervals;
use crate::{
    apply_synapses::{apply_synapses, build_apply_synapses_kernel},
    get_synapse_mask::get_synapse_mask,
    recount_accumulated_weights::{
        build_recount_accumulated_weights_kernel, recount_accumulated_weights,
    },
    spiral::{get_last_field, get_next_field},
    structures::{
        CompiledKernel, LayerParams, NetworkDumpDeserialize, NetworkDumpSerialize, SynapseMask,
        SynapseParams,
    },
};

struct ComputedParams {
    // number of fields in both layers
    field_count: usize,
    // number of neurons in one field
    field_size: usize,
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
    // acumulated weights of synapses from the first layer to the second layer
    accumulated_weights_1_to_2: Array2<f32>,
    // acumulated weights of synapses from the second layer to the first layer
    accumulated_weights_2_to_1: Array2<f32>,
    // distance weights of synapses from the first layer to the second layer
    distance_weights_1_to_2: Array2<f32>,
    // distance weights of synapses from the second layer to the first layer
    distance_weights_2_to_1: Array2<f32>,
    // compiled kernel for recount accumulated weights with opencl
    kernel_accumulated_weights: CompiledKernel,
    // compiled kernel for recount neurons and refract intervals with opencl
    kernel_synapses: CompiledKernel,
    // computed array of indexes of neurons in last field to receive prediction data
    last_field_indexes: Vec<usize>,
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
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    logger: Option<Box<dyn Logger>>,
}

fn get_neuron_index(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    layer_x: usize,
    layer_y: usize,
    neuron_in_field_x: usize,
    neuron_in_field_y: usize,
) -> usize {
    let layer_offset = computed_params.row_size * layer_y + computed_params.field_size * layer_x;
    let field_offset = layer_params.field_width * neuron_in_field_y + neuron_in_field_x;

    layer_offset + field_offset
}

fn get_last_field_indexes(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
) -> Vec<usize> {
    let mut res = vec![];

    let (last_field_x, last_field_y) = get_last_field(layer_params);

    for neuron_in_field_x in 0..layer_params.field_width {
        for neuron_in_field_y in 0..layer_params.field_height {
            res.push(get_neuron_index(
                layer_params,
                computed_params,
                last_field_x,
                last_field_y,
                neuron_in_field_x,
                neuron_in_field_y,
            ));
        }
    }

    res
}

fn get_neuron_coordinates(
    params: &LayerParams,
    layer_x: usize,
    layer_y: usize,
    neuron_in_field_x: usize,
    neuron_in_field_y: usize,
) -> (usize, usize) {
    let layer_x_offset = params.field_width * layer_x;
    let layer_y_offset = params.field_height * layer_y;

    let field_x_offset = neuron_in_field_x;
    let field_y_offset = neuron_in_field_y;

    (
        layer_x_offset + field_x_offset,
        layer_y_offset + field_y_offset,
    )
}

fn get_neuron_full_coordinates(
    params: &LayerParams,
    neuron_x: usize,
    neuron_y: usize,
) -> (usize, usize, usize, usize) {
    let neuron_in_field_x = neuron_x % params.field_width;
    let neuron_in_field_y = neuron_y % params.field_height;

    let layer_x = (neuron_x - neuron_in_field_x) / params.field_width;
    let layer_y = (neuron_y - neuron_in_field_y) / params.field_height;

    (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y)
}

fn get_neuron_index_by_coordinates(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    neuron_x: usize,
    neuron_y: usize,
) -> usize {
    let (layer_x, layer_y, neuron_in_field_x, neuron_in_field_y) =
        get_neuron_full_coordinates(layer_params, neuron_x, neuron_y);

    get_neuron_index(
        layer_params,
        computed_params,
        layer_x,
        layer_y,
        neuron_in_field_x,
        neuron_in_field_y,
    )
}

fn apply_mask(
    layer_params: &LayerParams,
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
                layer_params,
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
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    synapse_params: &SynapseParams,
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
    let layer_size = layer_params.field_width
        * layer_params.field_height
        * layer_params.layer_width
        * layer_params.layer_height;

    let mut distance_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut distance_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let mut accumulated_weights_1_to_2 = Array2::<f32>::zeros([layer_size, layer_size]);
    let mut accumulated_weights_2_to_1 = Array2::<f32>::zeros([layer_size, layer_size]);

    let (finish_x, finish_y) = get_last_field(layer_params);

    for layer_y in 0..layer_params.layer_height {
        for layer_x in 0..layer_params.layer_width {
            if layer_x != finish_x || layer_y != finish_y {
                let (layer_2_to_1_x, layer_2_to_1_y) =
                    get_next_field(layer_params, layer_x, layer_y);

                for neuron_in_field_y in 0..layer_params.field_height {
                    for neuron_in_field_x in 0..layer_params.field_width {
                        let neuron_index = get_neuron_index(
                            layer_params,
                            computed_params,
                            layer_x,
                            layer_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );
                        let neuron_2_to_1_index = get_neuron_index(
                            layer_params,
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
                            layer_params,
                            layer_x,
                            layer_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        apply_mask(
                            layer_params,
                            computed_params,
                            neuron_index,
                            &mut distance_weights_1_to_2,
                            mask,
                            x,
                            y,
                        );

                        let (x_2_to_1, y_2_to_1) = get_neuron_coordinates(
                            layer_params,
                            layer_2_to_1_x,
                            layer_2_to_1_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        apply_mask(
                            layer_params,
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
    }

    (
        distance_weights_1_to_2,
        distance_weights_2_to_1,
        accumulated_weights_1_to_2,
        accumulated_weights_2_to_1,
    )
}

fn get_computed_params(layer_params: &LayerParams) -> ComputedParams {
    let LayerParams {
        field_width,
        field_height,
        layer_width,
        layer_height,
    } = layer_params;

    let field_size = field_width * field_height;
    let row_size = field_size * layer_width;
    let row_width = field_width * layer_width;
    let column_height = field_height * layer_height;
    let field_count = layer_width * layer_height * 2;

    ComputedParams {
        field_size,
        field_count,
        row_size,
        row_width,
        column_height,
    }
}

fn get_layer_size(layer_params: &LayerParams, computed_params: &ComputedParams) -> usize {
    computed_params.field_size * layer_params.layer_width * layer_params.layer_height
}

#[derive(Debug)]
pub enum NetworkParseError {
    JSON(serde_json::Error),
    Gz(std::io::Error),
}

fn parse_json_dump(dump: &str) -> Result<NetworkDumpDeserialize, NetworkParseError> {
    match serde_json::from_str::<NetworkDumpDeserialize>(dump) {
        Ok(json) => Ok(json),
        Err(error) => Err(NetworkParseError::JSON(error)),
    }
}

impl Network {
    pub fn new(
        layer_params: LayerParams,
        synapse_params: SynapseParams,
        logger: Option<Box<dyn Logger>>,
    ) -> Self {
        let LayerParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
        } = layer_params;

        let field_size = field_width * field_height;

        let computed_params = get_computed_params(&layer_params);

        let layer_size = get_layer_size(&layer_params, &computed_params);

        let mask = get_synapse_mask(&synapse_params);

        let (
            distance_weights_1_to_2,
            distance_weights_2_to_1,
            accumulated_weights_1_to_2,
            accumulated_weights_2_to_1,
        ) = set_initial_connections(&layer_params, &computed_params, &synapse_params, &mask);

        let kernel_accumulated_weights =
            build_recount_accumulated_weights_kernel(layer_size).unwrap();
        let kernel_synapses = build_apply_synapses_kernel(layer_size).unwrap();

        let last_field_indexes = get_last_field_indexes(&layer_params, &computed_params);

        Network {
            accumulated_weights_1_to_2,
            accumulated_weights_2_to_1,
            computed_params,
            distance_weights_1_to_2,
            distance_weights_2_to_1,
            kernel_accumulated_weights,
            kernel_synapses,
            last_field_indexes,
            layer_width,
            layer_height,
            field_size,
            layer_size,
            neurons_1: Array1::<f32>::zeros(layer_size),
            neurons_2: Array1::<f32>::zeros(layer_size),
            refract_intervals_1: Array1::<u8>::zeros(layer_size),
            refract_intervals_2: Array1::<u8>::zeros(layer_size),
            layer_params,
            synapse_params,
            logger,
        }
    }

    pub fn from_json_dump(dump: &str) -> Result<Self, NetworkParseError> {
        let parsed_dump = parse_json_dump(dump)?;

        let LayerParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
        } = parsed_dump.layer_params;

        let field_size = field_width * field_height;

        let computed_params = get_computed_params(&parsed_dump.layer_params);

        let layer_size = get_layer_size(&parsed_dump.layer_params, &computed_params);

        let kernel_accumulated_weights =
            build_recount_accumulated_weights_kernel(layer_size).unwrap();
        let kernel_synapses = build_apply_synapses_kernel(layer_size).unwrap();

        let last_field_indexes =
            get_last_field_indexes(&parsed_dump.layer_params, &computed_params);

        let network = Network {
            accumulated_weights_1_to_2: parsed_dump.accumulated_weights_1_to_2,
            accumulated_weights_2_to_1: parsed_dump.accumulated_weights_2_to_1,
            computed_params,
            distance_weights_1_to_2: parsed_dump.distance_weights_1_to_2,
            distance_weights_2_to_1: parsed_dump.distance_weights_2_to_1,
            kernel_accumulated_weights,
            kernel_synapses,
            last_field_indexes,
            layer_width,
            layer_height,
            field_size,
            layer_size,
            neurons_1: parsed_dump.neurons_1,
            neurons_2: parsed_dump.neurons_2,
            refract_intervals_1: parsed_dump.refract_intervals_1,
            refract_intervals_2: parsed_dump.refract_intervals_2,
            layer_params: parsed_dump.layer_params,
            synapse_params: parsed_dump.synapse_params,
            logger: None,
        };

        Ok(network)
    }

    pub fn from_gzip_dump_str(dump: &str) -> Result<Self, NetworkParseError> {
        let bytes = dump.as_bytes();

        Network::from_gzip_dump_bytes(bytes)
    }

    pub fn from_gzip_dump_bytes(bytes: &[u8]) -> Result<Self, NetworkParseError> {
        let mut decoder = GzDecoder::new(bytes);
        let mut json = String::new();

        if let Err(gz_err) = decoder.read_to_string(&mut json) {
            return Err(NetworkParseError::Gz(gz_err));
        }

        Network::from_json_dump(&json)
    }

    pub fn get_json_dump(&self) -> String {
        let dump = NetworkDumpSerialize {
            accumulated_weights_1_to_2: &self.accumulated_weights_1_to_2,
            accumulated_weights_2_to_1: &self.accumulated_weights_2_to_1,
            distance_weights_1_to_2: &self.distance_weights_1_to_2,
            distance_weights_2_to_1: &self.distance_weights_2_to_1,
            neurons_1: &self.neurons_1,
            neurons_2: &self.neurons_2,
            refract_intervals_1: &self.refract_intervals_1,
            refract_intervals_2: &self.refract_intervals_2,
            layer_params: &self.layer_params,
            synapse_params: &self.synapse_params,
        };

        serde_json::to_string(&dump).unwrap()
    }

    pub fn get_gzip_dump(&self) -> Result<Vec<u8>, std::io::Error> {
        let json_dump = self.get_json_dump();

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(json_dump.as_bytes())?;
        let compressed_data = encoder.finish()?;

        Ok(compressed_data)
    }

    pub fn get_layer_params(&self) -> &LayerParams {
        &self.layer_params
    }

    fn shift(&mut self, bit_vec: &[bool]) {
        let data_len = bit_vec.len();

        if data_len > self.field_size {
            panic!("The length of the bit vec chunk should be less than or equal to the size of one field");
        }

        for (pos, value) in bit_vec.iter().enumerate() {
            self.neurons_1[[pos]] = if *value { 1.0 } else { 0.0 };
        }

        for pos in data_len..self.field_size {
            self.neurons_1[[pos]] = 0.0;
        }

        let next_neurons_2 = apply_synapses(
            &self.kernel_synapses,
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

        let next_accumulated_weights_1_to_2 = recount_accumulated_weights(
            &self.kernel_accumulated_weights,
            self.layer_size,
            &self.accumulated_weights_1_to_2,
            &self.neurons_1,
            &next_neurons_2,
            &self.refract_intervals_2,
            self.synapse_params.g_dec,
            self.synapse_params.g_inc,
            self.synapse_params.max_g,
            1,
            &mut self.logger,
        )
        .unwrap();

        let next_refract_intervals_1 = recount_refract_intervals(
            &self.neurons_1,
            &self.refract_intervals_1,
            &self.synapse_params.refract_interval,
        );

        self.neurons_2 = next_neurons_2;
        self.refract_intervals_1 = next_refract_intervals_1;
        self.accumulated_weights_1_to_2 = next_accumulated_weights_1_to_2;

        let next_neurons_1 = apply_synapses(
            &self.kernel_synapses,
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

        let next_accumulated_weights_2_to_1 = recount_accumulated_weights(
            &self.kernel_accumulated_weights,
            self.layer_size,
            &self.accumulated_weights_2_to_1,
            &self.neurons_2,
            &next_neurons_1,
            &self.refract_intervals_1,
            self.synapse_params.g_dec,
            self.synapse_params.g_inc,
            self.synapse_params.max_g,
            2,
            &mut self.logger,
        )
        .unwrap();

        let next_refract_intervals_2 = recount_refract_intervals(
            &self.neurons_2,
            &self.refract_intervals_2,
            &self.synapse_params.refract_interval,
        );

        self.neurons_1 = next_neurons_1;
        self.refract_intervals_2 = next_refract_intervals_2;
        self.accumulated_weights_2_to_1 = next_accumulated_weights_2_to_1;
    }

    fn split_signal(&self, bit_vec: &[bool]) -> (Vec<bool>, Option<Vec<bool>>) {
        let mut has_intersection = false;

        let mut apply_vec = vec![false; self.field_size];
        let mut rest_vec = vec![false; self.field_size];

        for (pos, value) in bit_vec.iter().enumerate() {
            if *value {
                if self.refract_intervals_1[[pos]] > 0 {
                    has_intersection = true;
                    rest_vec[pos] = true;
                } else {
                    apply_vec[pos] = true;
                }
            }
        }

        (
            apply_vec,
            if has_intersection {
                Some(rest_vec)
            } else {
                None
            },
        )
    }

    fn tick_not_intersected(&mut self, bit_vec: &[bool]) {
        self.shift(bit_vec);

        for _ in 0..self.synapse_params.signal_shift_interval {
            self.shift(&vec![]);
        }
    }

    pub fn tick(&mut self, bit_vec: &[bool]) {
        let (mut apply_vec, mut rest_vec) = self.split_signal(bit_vec);

        self.tick_not_intersected(&apply_vec);

        while rest_vec.is_some() {
            (apply_vec, rest_vec) = self.split_signal(&rest_vec.unwrap());
            self.tick_not_intersected(&apply_vec);
        }
    }

    pub fn push_data_binary(&mut self, bit_vec: &[bool]) {
        let data_len = bit_vec.len();

        let tick_count = if data_len % self.field_size == 0 {
            data_len / self.field_size
        } else {
            (data_len / self.field_size) + 1
        };

        for i in 0..tick_count {
            let start = i * self.field_size;
            let end = std::cmp::min(start + self.field_size, data_len);
            self.tick(&bit_vec[start..end]);
        }
    }

    pub fn predict(&mut self, bit_vec: &[bool]) -> Vec<bool> {
        let data_len = bit_vec.len();

        let tick_count = if data_len % self.field_size == 0 {
            data_len / self.field_size
        } else {
            (data_len / self.field_size) + 1
        };

        for i in 0..tick_count {
            let start = i * self.field_size;
            let end = std::cmp::min(start + self.field_size, data_len);
            self.tick(&bit_vec[start..end]);
        }

        for _ in tick_count..self.computed_params.field_count - 1 {
            self.tick(&vec![]);
        }

        let mut res = vec![];

        for _ in 0..tick_count {
            self.tick(&vec![]);

            let last_field_state = self.get_last_field_state();

            for neuron_value in last_field_state {
                res.push(neuron_value);
            }
        }

        res
    }

    pub fn get_last_field_state(&self) -> Vec<bool> {
        let mut res = vec![];

        for field_index in self.last_field_indexes.iter() {
            res.push(if self.neurons_2[*field_index] > 0.5 {
                true
            } else {
                false
            });
        }

        res
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
            for neuron_in_field_y in 0..self.layer_params.field_height {
                for layer_x in 0..self.layer_width {
                    for neuron_in_field_x in 0..self.layer_params.field_width {
                        let neuron_index = get_neuron_index(
                            &self.layer_params,
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
        (
            self.computed_params.row_width,
            self.computed_params.column_height,
        )
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
            &self.layer_params,
            &self.computed_params,
            layer_x,
            layer_y,
            neuron_in_field_x,
            neuron_in_field_y,
        );

        refract_intervals[[neuron_index]]
    }

    pub fn get_neuron_state(
        &self,
        layer_index: u8,
        layer_x: usize,
        layer_y: usize,
        neuron_in_field_x: usize,
        neuron_in_field_y: usize,
    ) -> f32 {
        let neurons = if layer_index == 1 {
            &self.neurons_1
        } else {
            &self.neurons_2
        };

        let neuron_index = get_neuron_index(
            &self.layer_params,
            &self.computed_params,
            layer_x,
            layer_y,
            neuron_in_field_x,
            neuron_in_field_y,
        );

        neurons[[neuron_index]]
    }

    pub fn get_neuron_full_coordinates(
        &self,
        neuron_x: usize,
        neuron_y: usize,
    ) -> (usize, usize, usize, usize) {
        get_neuron_full_coordinates(&self.layer_params, neuron_x, neuron_y)
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

        let neuron_index = get_neuron_index_by_coordinates(
            &self.layer_params,
            &self.computed_params,
            neuron_x,
            neuron_y,
        );

        for layer_y in 0..self.layer_height {
            for neuron_in_field_y in 0..self.layer_params.field_height {
                for layer_x in 0..self.layer_width {
                    for neuron_in_field_x in 0..self.layer_params.field_width {
                        let target_neuron_index = get_neuron_index(
                            &self.layer_params,
                            &self.computed_params,
                            layer_x,
                            layer_y,
                            neuron_in_field_x,
                            neuron_in_field_y,
                        );

                        let (target_x, target_y) = get_neuron_coordinates(
                            &self.layer_params,
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

        res
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

        self.get_neuron_weights(weights_layer, neuron_x, neuron_y)
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

        self.get_neuron_weights(weights_layer, neuron_x, neuron_y)
    }

    pub fn get_accumulated_weights_sum(&self, layer_index: u8) -> f32 {
        let weights_layer = if layer_index == 1 {
            &self.accumulated_weights_1_to_2
        } else {
            &self.accumulated_weights_2_to_1
        };

        weights_layer.sum()
    }
}

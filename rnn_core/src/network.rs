use std::io::prelude::*;
use std::io::Write;

use flate2::Compression;
use flate2::{read::GzDecoder, write::GzEncoder};
use ndarray::{Array1, Array2};

use crate::get_neuron_coordinates::get_neuron_coordinates;
use crate::get_neuron_full_coordinates::get_neuron_full_coordinates;
use crate::get_neuron_index::get_neuron_index;
use crate::get_neuron_index_by_coordinates::get_neuron_index_by_coordinates;
use crate::logger::Logger;
use crate::prediction::PredictionProcessing;
use crate::recount_refract_intervals::recount_refract_intervals;
use crate::set_initial_connections::set_initial_connections;
use crate::shift_signal::shift_signal;
use crate::structures::Action;
use crate::structures::ComputedParams;
use crate::LoggerEvent;
use crate::{
    apply_synapses::{apply_synapses, build_apply_synapses_kernel},
    get_synapse_mask::get_synapse_mask,
    recount_accumulated_weights::{
        build_recount_accumulated_weights_kernel, recount_accumulated_weights,
    },
    spiral::get_last_field,
    structures::{
        CompiledKernel, LayerParams, NetworkDumpDeserialize, NetworkDumpSerialize, SynapseParams,
    },
};

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
    neurons_1: Array1<u8>,
    // neuron states at the second layer
    neurons_2: Array1<u8>,
    // timeouts of neuron refract states of the first layer
    refract_intervals_1: Array1<u8>,
    // timeouts of neuron refract states of the second layer
    refract_intervals_2: Array1<u8>,
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    logger: Option<Box<dyn Logger>>,
    action_queue: Vec<Action>,
    signal_buffer: Vec<Vec<bool>>,
    prediction: Option<PredictionProcessing>,
}

fn get_last_field_indexes(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
) -> Vec<usize> {
    let mut res = vec![];

    let (last_field_x, last_field_y) = get_last_field(layer_params);

    for neuron_in_field_y in 0..layer_params.field_height {
        for neuron_in_field_x in 0..layer_params.field_width {
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

fn get_computed_params(
    layer_params: &LayerParams,
    synapse_params: &SynapseParams,
) -> ComputedParams {
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
    let field_count = layer_width * layer_height;
    let single_signal_shifts = 1
        + synapse_params
            .signal_copy_shifts
            .as_ref()
            .map_or(0, |signal_copy_shifts| signal_copy_shifts.len())
        + synapse_params.signal_shift_interval as usize;
    let prediction_rest_shifts = if field_count > single_signal_shifts {
        field_count - single_signal_shifts
    } else {
        0
    };

    let excited_neurons_limit =
        ((field_size * field_count) as f32 * synapse_params.excite_neuron_limit) as usize;

    ComputedParams {
        field_size,
        field_count,
        row_size,
        row_width,
        column_height,
        prediction_rest_shifts,
        excited_neurons_limit,
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

        let computed_params = get_computed_params(&layer_params, &synapse_params);

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
            neurons_1: Array1::<u8>::zeros(layer_size),
            neurons_2: Array1::<u8>::zeros(layer_size),
            refract_intervals_1: Array1::<u8>::zeros(layer_size),
            refract_intervals_2: Array1::<u8>::zeros(layer_size),
            layer_params,
            synapse_params,
            logger,
            action_queue: vec![],
            signal_buffer: vec![],
            prediction: None,
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

        let computed_params =
            get_computed_params(&parsed_dump.layer_params, &parsed_dump.synapse_params);

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
            action_queue: vec![],
            signal_buffer: vec![],
            prediction: None,
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

    fn input_signal(&mut self, bit_vec: &[bool]) {
        let data_len = bit_vec.len();

        if data_len > self.field_size {
            panic!("The length of the bit vec chunk should be less than or equal to the size of one field");
        }

        if let Some(logger) = &mut self.logger {
            logger.log_event(LoggerEvent::Input(bit_vec.to_vec()));
        }

        for (pos, value) in bit_vec.iter().enumerate() {
            if *value && self.refract_intervals_1[[pos]] == 0 {
                self.neurons_1[[pos]] = 1;
            }
        }
    }

    fn shift_1_to_2(&mut self) {
        apply_synapses(
            &self.kernel_synapses,
            self.layer_size,
            &self.accumulated_weights_1_to_2,
            &self.distance_weights_1_to_2,
            &self.neurons_1,
            &mut self.neurons_2,
            &self.refract_intervals_2,
            self.synapse_params.refract_interval,
            self.synapse_params.threshold,
            self.synapse_params.gamma_inc,
            self.synapse_params.gamma_dec,
            0.0,
            self.computed_params.excited_neurons_limit,
        )
        .unwrap();

        recount_accumulated_weights(
            &self.kernel_accumulated_weights,
            self.layer_size,
            &mut self.accumulated_weights_1_to_2,
            &self.neurons_1,
            &self.neurons_2,
            &self.refract_intervals_2,
            self.synapse_params.g_dec,
            self.synapse_params.g_inc,
            self.synapse_params.min_g,
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

        self.refract_intervals_1 = next_refract_intervals_1;

        if self.logger.is_some() {
            let total_2 = self.get_accumulated_weights_sum(2);
            self.logger
                .as_mut()
                .unwrap()
                .log_event(LoggerEvent::LayerTotalWeight(2, total_2));
        }
    }

    fn shift_2_to_1(&mut self) {
        apply_synapses(
            &self.kernel_synapses,
            self.layer_size,
            &self.accumulated_weights_2_to_1,
            &self.distance_weights_2_to_1,
            &self.neurons_2,
            &mut self.neurons_1,
            &self.refract_intervals_1,
            self.synapse_params.refract_interval,
            self.synapse_params.threshold,
            self.synapse_params.gamma_inc,
            self.synapse_params.gamma_dec,
            self.synapse_params.g_0,
            self.computed_params.excited_neurons_limit,
        )
        .unwrap();

        recount_accumulated_weights(
            &self.kernel_accumulated_weights,
            self.layer_size,
            &mut self.accumulated_weights_2_to_1,
            &self.neurons_2,
            &self.neurons_1,
            &self.refract_intervals_1,
            self.synapse_params.g_dec,
            self.synapse_params.g_inc,
            self.synapse_params.min_g,
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

        self.refract_intervals_2 = next_refract_intervals_2;

        if self.logger.is_some() {
            let total_1 = self.get_accumulated_weights_sum(1);
            self.logger
                .as_mut()
                .unwrap()
                .log_event(LoggerEvent::LayerTotalWeight(1, total_1));
        }
    }

    fn shift(&mut self, bit_vec: &[bool]) {
        self.input_signal(bit_vec);
        self.shift_1_to_2();
        self.shift_2_to_1();
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

    /**
     * Set the values of neurons to the input field and make shifts before setting the second part
     *
     * The values are guaranteed not to overlap with the refractive neurons
     */
    fn tick_not_intersected(
        &mut self,
        bit_vec: &[bool],
        _prediction: &mut Option<PredictionProcessing>,
    ) {
        self.shift(bit_vec);

        /* if let Some(prediction) = prediction {
            prediction.add_tick_split();

            if prediction.should_read() {
                prediction.read(&self.get_last_field_state());
            }
        } */

        for _ in 0..self.synapse_params.signal_shift_interval {
            self.shift(&vec![]);

            /* if let Some(prediction) = prediction {
                prediction.shift();

                if prediction.should_read() {
                    prediction.read(&self.get_last_field_state());
                }
            } */
        }
    }

    /**
     * Set the values of neurons to the input field and make shifts before setting the second part
     *
     * If there are values that overlap with the refractive neurons, make a tick without them and recursively call this method with that values
     */
    pub fn tick(
        &mut self,
        bit_vec: &[bool],
        prediction: &mut Option<PredictionProcessing>,
        apply_rest: bool,
    ) {
        let (mut apply_vec, mut rest_vec) = self.split_signal(bit_vec);

        self.tick_not_intersected(&apply_vec, prediction);

        if !apply_rest {
            return;
        }

        let mut counter = 0u8;

        let limit = self.synapse_params.signal_rest_shift_limit.unwrap_or(255);

        while rest_vec.is_some() && counter < limit {
            (apply_vec, rest_vec) = self.split_signal(&rest_vec.unwrap());
            self.tick_not_intersected(&apply_vec, prediction);
            counter += 1;
        }
    }

    /**
     * Split signal into frames and apply them immediately
     */
    pub fn push_data_and_apply(&mut self, bit_vec: &[bool], prediction_depth: usize) {
        self.push_data_binary(bit_vec, prediction_depth);
        self.apply_buffer();
    }

    /**
     * Split signal into frames and push them to buffer
     */
    pub fn push_data_binary(&mut self, bit_vec: &[bool], _prediction_depth: usize) {
        let data_len = bit_vec.len();
        let field_size = self.field_size;
        let tick_count = self.get_tick_count(bit_vec);

        for i in 0..tick_count {
            let start = i * self.field_size;
            let end = std::cmp::min(start + field_size, data_len);

            if let Some(prediction) = &mut self.prediction {
                prediction.add_tick(i);
            }

            self.push_to_buffer(bit_vec[start..end].to_vec());
        }
    }

    fn get_tick_count(&self, bit_vec: &[bool]) -> usize {
        let data_len = bit_vec.len();
        let field_size = self.field_size;

        if data_len == 0 {
            return 1;
        }

        if data_len % field_size == 0 {
            return data_len / field_size;
        }

        (data_len / field_size) + 1
    }

    pub fn predict(&mut self, bit_vec: &[bool], prediction_depth: usize) -> Vec<bool> {
        let tick_count = self.get_tick_count(bit_vec);

        self.prediction = Some(PredictionProcessing::new(
            tick_count,
            self.computed_params.field_size,
            self.computed_params.field_count,
        ));

        self.push_data_and_apply(bit_vec, prediction_depth);

        self.prediction.as_ref().unwrap().get_prediction()
    }

    /**
     * Set all the values of neurons and refract intervals to 0
     */
    fn _clean_neurons(&mut self) {
        let layer_size = self.layer_size;

        self.neurons_1 = Array1::<u8>::zeros(layer_size);
        self.neurons_2 = Array1::<u8>::zeros(layer_size);
        self.refract_intervals_1 = Array1::<u8>::zeros(layer_size);
        self.refract_intervals_2 = Array1::<u8>::zeros(layer_size);
    }

    pub fn _predict(&mut self, bit_vec: &[bool]) -> Vec<bool> {
        let data_len = bit_vec.len();

        let tick_count = if data_len % self.field_size == 0 {
            data_len / self.field_size
        } else {
            (data_len / self.field_size) + 1
        };

        let mut prediction = Some(PredictionProcessing::new(
            tick_count,
            self.computed_params.field_size,
            self.computed_params.field_count,
        ));

        for i in 0..tick_count {
            let start = i * self.field_size;
            let end = std::cmp::min(start + self.field_size, data_len);

            prediction.as_mut().unwrap().add_tick(i);

            self.tick(&bit_vec[start..end], &mut prediction, true);
        }

        let prediction = prediction.as_mut().unwrap();

        while !prediction.is_finished() {
            for _ in 0..=self.synapse_params.signal_shift_interval {
                self.shift(&vec![]);

                prediction.shift();

                if prediction.should_read() {
                    prediction.read(&self.get_last_field_state());
                }
            }
        }

        prediction.get_prediction()
    }

    pub fn get_last_field_state(&self) -> Vec<u8> {
        let mut res: Vec<u8> = vec![];

        for field_index in self.last_field_indexes.iter() {
            res.push(self.neurons_2[[*field_index]]);
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

    fn print_state(&self, layer: &Array1<u8>) {
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

                        print!("{} ", if layer[[neuron_index]] > 0 { "+" } else { "." });
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
    ) -> u8 {
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

    fn push_shift(&mut self, bit_vec: Vec<bool>, is_source: bool) {
        self.action_queue
            .push(Action::InputSignal(bit_vec, is_source));
        self.action_queue.push(Action::EmptyShift1to2);
        self.action_queue.push(Action::EmptyShift2to1);
    }

    fn push_empty_shift(&mut self) {
        self.action_queue.push(Action::EmptyShift1to2);
        self.action_queue.push(Action::EmptyShift2to1);
    }

    fn push_shifted_signals(&mut self, bit_vec: &[bool]) {
        if let Some(shifts) = &self.synapse_params.signal_copy_shifts {
            let shifted_signals: Vec<Vec<bool>> = shifts
                .into_iter()
                .map(|shift| {
                    return shift_signal(&bit_vec, self.field_size, &self.layer_params, shift);
                })
                .collect();

            for shifted_signal in shifted_signals.into_iter() {
                self.push_shift(shifted_signal, false);
            }
        }
    }

    fn apply_signal(&mut self, bit_vec: &[bool], counter: &u8) {
        let (apply_vec, rest) = self.split_signal(bit_vec);

        self.push_shift(apply_vec, true);

        self.push_shifted_signals(bit_vec);

        for _ in 0..self.synapse_params.signal_shift_interval {
            self.push_empty_shift();
        }

        let is_finish =
            rest.is_none() || *counter < self.synapse_params.signal_rest_shift_limit.unwrap_or(255);

        if is_finish {
            if self
                .prediction
                .as_ref()
                .map_or(false, |prediction| prediction.is_all_ticks_added())
            {
                for _ in 0..self.computed_params.prediction_rest_shifts {
                    self.push_empty_shift();
                }
            }

            return;
        }

        self.action_queue
            .push(Action::ApplyRest(rest.unwrap(), *counter + 1));
    }

    fn apply_action(&mut self, action: &Action) {
        match action {
            Action::ApplyRest(bit_vec, counter) => {
                self.apply_signal(bit_vec, counter);
            }
            Action::InputSignal(bit_vec, is_source) => {
                self.input_signal(bit_vec);

                if *is_source {
                    if let Some(prediction) = &mut self.prediction {
                        prediction.add_tick_split();
                    }
                }
            }
            Action::EmptyShift1to2 => {
                self.shift_1_to_2();

                let should_read = self
                    .prediction
                    .as_mut()
                    .map_or(false, |prediction| prediction.should_read());

                if should_read {
                    let last_field_state = self.get_last_field_state();

                    self.prediction.as_mut().unwrap().read(&last_field_state);
                }
            }
            Action::EmptyShift2to1 => {
                self.shift_2_to_1();

                if let Some(prediction) = &mut self.prediction {
                    prediction.shift();
                }
            }
        }
    }

    pub fn apply_next_action(&mut self) {
        if self.action_queue.len() == 0 {
            return;
        }

        let action = self.action_queue.remove(0);
        self.apply_action(&action);
    }

    pub fn has_next_action(&self) -> bool {
        self.action_queue.len() > 0
    }

    pub fn apply_queue(&mut self) {
        while self.has_next_action() {
            self.apply_next_action();
        }
    }

    fn push_to_buffer(&mut self, signal: Vec<bool>) {
        self.signal_buffer.push(signal);
    }

    pub fn read_from_buffer(&mut self) {
        if self.signal_buffer.len() == 0 {
            return;
        }

        let signal = self.signal_buffer.remove(0);

        self.apply_signal(&signal, &0);
    }

    pub fn has_signal_in_buffer(&self) -> bool {
        self.signal_buffer.len() > 0
    }

    pub fn apply_buffer(&mut self) {
        while self.has_signal_in_buffer() {
            self.read_from_buffer();
            self.apply_queue();
        }
    }
}

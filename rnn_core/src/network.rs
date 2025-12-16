use std::collections::HashMap;
use std::collections::HashSet;
use std::io::prelude::*;
use std::io::Write;

use flate2::Compression;
use flate2::{read::GzDecoder, write::GzEncoder};
use ocl::Buffer;

use crate::apply_synapses_cpu_fallback::apply_synapses_cpu_fallback;
use crate::get_neuron_coordinates::get_neuron_coordinates;
use crate::get_neuron_full_coordinates::get_neuron_full_coordinates;
use crate::get_neuron_index::get_neuron_index;
use crate::get_neuron_index_by_coordinates::get_neuron_index_by_coordinates;
use crate::logger::Logger;
use crate::prediction::PredictionProcessing;
use crate::recount_refract_intervals::recount_refract_intervals;
use crate::set_initial_connections::set_initial_connections;
use crate::spiral::get_output_field;
use crate::structures::Action;
use crate::structures::ComputedParams;
use crate::structures::InitialConnections;
use crate::structures::InputPhase;
use crate::structures::PartitionPayloadByIndex;
use crate::LoggerEvent;
use crate::{
    apply_synapses::{apply_synapses, build_apply_synapses_kernel},
    get_synapse_mask::get_synapse_mask,
    structures::{
        CompiledKernel, LayerParams, NetworkDumpDeserialize, NetworkDumpSerialize, SynapseParams,
    },
};

// TO DO: bit-vec / bitfield
pub struct Network {
    computed_params: ComputedParams,
    offsets_1_to_2: Vec<u64>,
    buffer_offsets_1_to_2: Buffer<u64>,
    offsets_2_to_1: Vec<u64>,
    buffer_offsets_2_to_1: Buffer<u64>,
    restore_offsets_1_to_2: Vec<u64>,
    buffer_restore_offsets_1_to_2: Buffer<u64>,
    restore_synapses_1_to_2: Vec<f32>,
    buffer_restore_synapses_1_to_2: Buffer<f32>,
    // acumulated weights of synapses from the first layer to the second layer
    forward_synapses_1_to_2: Vec<f32>,
    buffer_forward_synapses_1_to_2: Buffer<f32>,
    forward_distance_weights: Vec<f32>,
    buffer_forward_distance_weights: Buffer<f32>,
    restore_distance_weights_1_to_2: Vec<f32>,
    buffer_restore_distance_weights_1_to_2: Buffer<f32>,
    // acumulated weights of synapses from the second layer to the first layer
    forward_synapses_2_to_1: Vec<f32>,
    buffer_forward_synapses_2_to_1: Buffer<f32>,
    // compiled kernel for recount neurons and refract intervals with opencl
    kernel_synapses: CompiledKernel,
    output_field_index: usize,
    // computed array of indexes of neurons in last field to receive prediction data
    output_field_neuron_indexes: Vec<usize>,
    layer_width: usize,
    layer_height: usize,
    // number of neurons
    field_size: usize,
    // number of neurons in one layer
    layer_size: usize,
    // neuron states at the first layer
    neurons_1: Vec<u8>,
    // neuron states at the second layer
    neurons_2: Vec<u8>,
    // timeouts of neuron refract states of the first layer
    refract_intervals_1: Vec<u8>,
    // timeouts of neuron refract states of the second layer
    refract_intervals_2: Vec<u8>,
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    logger: Option<Box<dyn Logger>>,
    action_queue: Vec<Action>,
    signal_buffer: Vec<Vec<bool>>,
    prediction: Option<PredictionProcessing>,
    input_phase: InputPhase,
    cpu_fallback_signals_buffer: Vec<f32>,
    enable_cpu_fallback: bool,
}

fn get_output_field_index(synapse_params: &SynapseParams) -> usize {
    synapse_params.signal_shift_interval as usize
}

fn get_output_field_neuron_indexes(
    layer_params: &LayerParams,
    computed_params: &ComputedParams,
    output_field_index: usize,
) -> Vec<usize> {
    let mut res = vec![];

    let (output_field_x, output_field_y) = get_output_field(layer_params, output_field_index);

    for neuron_in_field_y in 0..layer_params.field_height {
        for neuron_in_field_x in 0..layer_params.field_width {
            res.push(get_neuron_index(
                layer_params,
                computed_params,
                output_field_x,
                output_field_y,
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
        ..
    } = layer_params;

    let field_size = field_width * field_height;
    let row_size = field_size * layer_width;
    let layer_size = row_size * layer_height;
    let row_width = field_width * layer_width;
    let column_height = field_height * layer_height;
    let field_count = layer_width * layer_height;

    // uncomment to read from the last field

    /* let single_signal_shifts = 1 + synapse_params.signal_shift_interval as usize;

    let prediction_rest_shifts = if field_count > single_signal_shifts {
        field_count - single_signal_shifts
    } else {
        0
    }; */

    let prediction_rest_shifts = 0;

    let max_excited_neurons_number =
        field_size * field_count / (1 + synapse_params.signal_shift_interval as usize);

    let excited_neurons_limit =
        (max_excited_neurons_number as f32 * synapse_params.excite_neuron_limit) as usize;

    let map_index_to_partition_data = match &layer_params.partitions {
        Some(partitions) => {
            let mut index = 0;
            let mut result = HashMap::new();

            for (partition_index, partition) in partitions.iter().enumerate() {
                let correlate_only = match &partition.correlate_only {
                    Some(correlate_only) => {
                        let hash_set: HashSet<usize> = correlate_only.clone().into_iter().collect();

                        Some(hash_set)
                    }
                    _ => None,
                };

                let no_correlate = match &partition.no_correlate {
                    Some(no_correlate) => {
                        let hash_set: HashSet<usize> = no_correlate.clone().into_iter().collect();

                        Some(hash_set)
                    }
                    _ => None,
                };

                for _ in 0..partition.size * 2 {
                    result.insert(
                        index,
                        PartitionPayloadByIndex {
                            partition_index,
                            correlate_only: correlate_only.clone(),
                            correlate_only_self: partition.correlate_only_self,
                            no_correlate: no_correlate.clone(),
                        },
                    );
                    index += 1;
                }
            }

            Some(result)
        }
        _ => None,
    };

    ComputedParams {
        field_size,
        field_count,
        layer_size,
        row_size,
        row_width,
        column_height,
        prediction_rest_shifts,
        excited_neurons_limit,
        max_excited_neurons_number: excited_neurons_limit as f32,
        map_index_to_partition_data,
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
            ..
        } = layer_params;

        let field_size = field_width * field_height;

        let computed_params = get_computed_params(&layer_params, &synapse_params);

        let layer_size = get_layer_size(&layer_params, &computed_params);

        let mask = get_synapse_mask(&synapse_params);

        let InitialConnections {
            forward_synapses_1_to_2,
            forward_synapses_2_to_1,
            offsets_1_to_2,
            offsets_2_to_1,
            restore_offsets_1_to_2,
            restore_synapses_1_to_2,
            forward_distance_weights,
            restore_distance_weights_1_to_2,
        } = set_initial_connections(&layer_params, &computed_params, &synapse_params, &mask);

        let kernel_synapses = build_apply_synapses_kernel(layer_size).unwrap();

        let buffer_offsets_1_to_2 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(offsets_1_to_2.len())
            .copy_host_slice(offsets_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_offsets_2_to_1 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(offsets_2_to_1.len())
            .copy_host_slice(offsets_2_to_1.as_slice())
            .build()
            .unwrap();

        let can_restore = restore_offsets_1_to_2.len() > 0;

        let buffer_restore_offsets_1_to_2 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(if can_restore {
                restore_offsets_1_to_2.len()
            } else {
                1
            })
            .copy_host_slice(if can_restore {
                restore_offsets_1_to_2.as_slice()
            } else {
                &[0]
            })
            .build()
            .unwrap();

        let buffer_restore_synapses_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(if can_restore {
                restore_synapses_1_to_2.len()
            } else {
                1
            })
            .copy_host_slice(if can_restore {
                restore_synapses_1_to_2.as_slice()
            } else {
                &[0.0_f32]
            })
            .build()
            .unwrap();

        let buffer_forward_distance_weights = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(forward_distance_weights.len())
            .copy_host_slice(forward_distance_weights.as_slice())
            .build()
            .unwrap();

        let buffer_restore_distance_weights_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(if can_restore {
                restore_distance_weights_1_to_2.len()
            } else {
                1
            })
            .copy_host_slice(if can_restore {
                restore_distance_weights_1_to_2.as_slice()
            } else {
                &[0.0_f32]
            })
            .build()
            .unwrap();

        let buffer_forward_synapses_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(forward_synapses_1_to_2.len())
            .copy_host_slice(forward_synapses_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_forward_synapses_2_to_1 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(forward_synapses_2_to_1.len())
            .copy_host_slice(forward_synapses_2_to_1.as_slice())
            .build()
            .unwrap();

        let output_field_index = get_output_field_index(&synapse_params);

        let output_field_neuron_indexes =
            get_output_field_neuron_indexes(&layer_params, &computed_params, output_field_index);

        Network {
            offsets_1_to_2,
            buffer_offsets_1_to_2,
            offsets_2_to_1,
            buffer_offsets_2_to_1,
            restore_offsets_1_to_2,
            buffer_restore_offsets_1_to_2,
            restore_synapses_1_to_2,
            buffer_restore_synapses_1_to_2,
            forward_synapses_1_to_2,
            buffer_forward_synapses_1_to_2,
            forward_synapses_2_to_1,
            buffer_forward_synapses_2_to_1,
            forward_distance_weights,
            buffer_forward_distance_weights,
            restore_distance_weights_1_to_2,
            buffer_restore_distance_weights_1_to_2,
            computed_params,
            kernel_synapses,
            output_field_index,
            output_field_neuron_indexes,
            layer_width,
            layer_height,
            field_size,
            layer_size,
            neurons_1: vec![0u8; layer_size],
            neurons_2: vec![0u8; layer_size],
            refract_intervals_1: vec![0u8; layer_size],
            refract_intervals_2: vec![0u8; layer_size],
            layer_params,
            synapse_params,
            logger,
            action_queue: vec![],
            signal_buffer: vec![],
            prediction: None,
            input_phase: InputPhase::Even,
            cpu_fallback_signals_buffer: vec![0f32; layer_size],
            enable_cpu_fallback: option_env!("CPU_FALLBACK").is_some(),
        }
    }

    pub fn from_json_dump(dump: &str) -> Result<Self, NetworkParseError> {
        let parsed_dump = parse_json_dump(dump)?;

        let LayerParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
            ..
        } = parsed_dump.layer_params;

        let field_size = field_width * field_height;

        let computed_params =
            get_computed_params(&parsed_dump.layer_params, &parsed_dump.synapse_params);

        let layer_size = get_layer_size(&parsed_dump.layer_params, &computed_params);

        let kernel_synapses = build_apply_synapses_kernel(layer_size).unwrap();

        let output_field_index = get_output_field_index(&parsed_dump.synapse_params);

        let output_field_neuron_indexes = get_output_field_neuron_indexes(
            &parsed_dump.layer_params,
            &computed_params,
            output_field_index,
        );

        let buffer_offsets_1_to_2 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.offsets_1_to_2.len())
            .copy_host_slice(parsed_dump.offsets_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_offsets_2_to_1 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.offsets_2_to_1.len())
            .copy_host_slice(parsed_dump.offsets_2_to_1.as_slice())
            .build()
            .unwrap();

        let buffer_restore_offsets_1_to_2 = Buffer::<u64>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.restore_offsets_1_to_2.len())
            .copy_host_slice(parsed_dump.restore_offsets_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_restore_synapses_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.restore_synapses_1_to_2.len())
            .copy_host_slice(parsed_dump.restore_synapses_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_forward_distance_weights = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.forward_distance_weights.len())
            .copy_host_slice(parsed_dump.forward_distance_weights.as_slice())
            .build()
            .unwrap();

        let buffer_restore_distance_weights_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_ONLY)
            .len(parsed_dump.restore_distance_weights_1_to_2.len())
            .copy_host_slice(parsed_dump.restore_distance_weights_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_forward_synapses_1_to_2 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(parsed_dump.forward_synapses_1_to_2.len())
            .copy_host_slice(parsed_dump.forward_synapses_1_to_2.as_slice())
            .build()
            .unwrap();

        let buffer_forward_synapses_2_to_1 = Buffer::<f32>::builder()
            .queue(kernel_synapses.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(parsed_dump.forward_synapses_2_to_1.len())
            .copy_host_slice(parsed_dump.forward_synapses_2_to_1.as_slice())
            .build()
            .unwrap();

        let network = Network {
            offsets_1_to_2: parsed_dump.offsets_1_to_2,
            buffer_offsets_1_to_2,
            offsets_2_to_1: parsed_dump.offsets_2_to_1,
            buffer_offsets_2_to_1,
            restore_offsets_1_to_2: parsed_dump.restore_offsets_1_to_2,
            buffer_restore_offsets_1_to_2,
            restore_synapses_1_to_2: parsed_dump.restore_synapses_1_to_2,
            buffer_restore_synapses_1_to_2,
            forward_synapses_1_to_2: parsed_dump.forward_synapses_1_to_2,
            buffer_forward_synapses_1_to_2,
            forward_synapses_2_to_1: parsed_dump.forward_synapses_2_to_1,
            buffer_forward_synapses_2_to_1,
            computed_params,
            forward_distance_weights: parsed_dump.forward_distance_weights,
            buffer_forward_distance_weights,
            restore_distance_weights_1_to_2: parsed_dump.restore_distance_weights_1_to_2,
            buffer_restore_distance_weights_1_to_2,
            kernel_synapses,
            output_field_index,
            output_field_neuron_indexes,
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
            input_phase: parsed_dump.input_phase,
            cpu_fallback_signals_buffer: vec![0f32; layer_size],
            enable_cpu_fallback: option_env!("CPU_FALLBACK").is_some(),
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
            forward_synapses_1_to_2: &self.forward_synapses_1_to_2,
            forward_synapses_2_to_1: &self.forward_synapses_2_to_1,
            forward_distance_weights: &self.forward_distance_weights,
            restore_distance_weights_1_to_2: &self.restore_distance_weights_1_to_2,
            neurons_1: &self.neurons_1,
            neurons_2: &self.neurons_2,
            refract_intervals_1: &self.refract_intervals_1,
            refract_intervals_2: &self.refract_intervals_2,
            layer_params: &self.layer_params,
            synapse_params: &self.synapse_params,
            input_phase: &self.input_phase,
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

    fn input_signal(&mut self, bit_vec: &[bool], input_phase: InputPhase) {
        let data_len = bit_vec.len();

        if data_len > self.field_size {
            panic!("The length of the bit vec chunk should be less than or equal to the size of one field");
        }

        if let Some(logger) = &mut self.logger {
            logger.log_event(LoggerEvent::Input(bit_vec.to_vec()));
        }

        /* let mut positive = 0;
        for value in bit_vec {
            print!("{}", if *value {"+"} else {"."} );

            if *value {
                positive += 1;
            }
        }
        println!("{}", positive); */

        for (pos, value) in bit_vec.iter().enumerate() {
            let neuron_index = match input_phase {
                InputPhase::Even => pos * 2,
                InputPhase::Odd => pos * 2 + 1,
            };

            if *value && self.refract_intervals_1[neuron_index] == 0 {
                self.neurons_1[neuron_index] = 1;
            }
        }
    }

    fn get_threshold(&self, neurons_from: &Vec<u8>) -> f32 {
        if self.prediction.is_some() {
            let excided_neurons_count = neurons_from.iter().filter(|&&x| x != 0).count() as f32;

            let excited_percentage =
                excided_neurons_count / self.computed_params.max_excited_neurons_number;

            let threshold = self.synapse_params.threshold_predict_min
                + (self.synapse_params.threshold_predict_max
                    - self.synapse_params.threshold_predict_min)
                    * excited_percentage;

            // println!("{} {} {}", excided_neurons_count, self.computed_params.max_excited_neurons_number, threshold);

            threshold
        } else {
            1.0
        }
    }

    fn shift_1_to_2(&mut self) {
        let threshold = self.get_threshold(&self.neurons_1);

        if self.enable_cpu_fallback {
            apply_synapses_cpu_fallback(
                self.prediction.is_some(),
                self.layer_size,
                &mut self.forward_synapses_1_to_2,
                &self.forward_distance_weights,
                Some(&self.restore_distance_weights_1_to_2),
                &self.offsets_1_to_2,
                Some(&self.restore_offsets_1_to_2),
                Some(&mut self.restore_synapses_1_to_2),
                &self.restore_offsets_1_to_2.len(),
                &mut self.neurons_1,
                &mut self.neurons_2,
                &self.refract_intervals_2,
                &self.layer_params,
                &self.synapse_params,
                &self.computed_params,
                threshold,
                1,
                &mut self.cpu_fallback_signals_buffer,
            )
            .unwrap();
        } else {
            apply_synapses(
                &self.kernel_synapses,
                self.prediction.is_some(),
                self.layer_size,
                &self.buffer_forward_synapses_1_to_2,
                &self.buffer_forward_distance_weights,
                Some(&self.buffer_restore_distance_weights_1_to_2),
                &self.buffer_offsets_1_to_2,
                Some(&self.buffer_restore_offsets_1_to_2),
                Some(&self.buffer_restore_synapses_1_to_2),
                &self.restore_offsets_1_to_2.len(),
                &self.neurons_1,
                &mut self.neurons_2,
                &self.refract_intervals_2,
                &self.layer_params,
                &self.synapse_params,
                &self.computed_params,
                threshold,
                1,
                &mut self.logger,
            )
            .unwrap();
        }

        recount_refract_intervals(
            &self.neurons_1,
            &mut self.refract_intervals_1,
            &self.synapse_params.refract_interval,
        );

        if self.logger.is_some() {
            let total_2 = self.get_forward_synapses_sum(2);
            self.logger
                .as_mut()
                .unwrap()
                .log_event(LoggerEvent::LayerTotalWeight(2, total_2));
        }
    }

    fn shift_2_to_1(&mut self) {
        let threshold = self.get_threshold(&self.neurons_2);

        if self.enable_cpu_fallback {
            apply_synapses_cpu_fallback(
                self.prediction.is_some(),
                self.layer_size,
                &mut self.forward_synapses_2_to_1,
                &self.forward_distance_weights,
                None,
                &self.offsets_2_to_1,
                None,
                None,
                &0,
                &self.neurons_2,
                &mut self.neurons_1,
                &self.refract_intervals_1,
                &self.layer_params,
                &self.synapse_params,
                &self.computed_params,
                threshold,
                2,
                &mut self.cpu_fallback_signals_buffer,
            )
            .unwrap();
        } else {
            apply_synapses(
                &self.kernel_synapses,
                self.prediction.is_some(),
                self.layer_size,
                &self.buffer_forward_synapses_2_to_1,
                &self.buffer_forward_distance_weights,
                None,
                &self.buffer_offsets_2_to_1,
                None,
                None,
                &0,
                &self.neurons_2,
                &mut self.neurons_1,
                &self.refract_intervals_1,
                &self.layer_params,
                &self.synapse_params,
                &self.computed_params,
                threshold,
                2,
                &mut self.logger,
            )
            .unwrap();
        }

        recount_refract_intervals(
            &self.neurons_2,
            &mut self.refract_intervals_2,
            &self.synapse_params.refract_interval,
        );

        if self.logger.is_some() {
            let total_1 = self.get_forward_synapses_sum(1);
            self.logger
                .as_mut()
                .unwrap()
                .log_event(LoggerEvent::LayerTotalWeight(1, total_1));
        }
    }

    fn split_signal(&self, bit_vec: &[bool]) -> (Vec<bool>, Option<Vec<bool>>) {
        let mut has_intersection = false;

        let half_field_size = self.field_size / 2;

        let mut apply_vec = vec![false; half_field_size];
        let mut rest_vec = vec![false; half_field_size];

        for (pos, value) in bit_vec.iter().enumerate() {
            let neuron_index = match self.input_phase {
                InputPhase::Even => pos * 2,
                InputPhase::Odd => pos * 2 + 1,
            };

            if *value {
                if self.refract_intervals_1[neuron_index] > 0 {
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
     * Split signal into frames and apply them immediately
     */
    pub fn push_data_and_apply(&mut self, bit_vec: &[bool], prediction_depth: usize) {
        // uncomment to enable logging
        /* for value in bit_vec {
            print!("{}", if *value { "+" } else { "." });
        }

        println!(); */
        self.push_data_binary(bit_vec, prediction_depth);
        self.apply_buffer();
    }

    /**
     * Split signal into frames and push them to buffer
     */
    pub fn push_data_binary(&mut self, bit_vec: &[bool], prediction_depth: usize) {
        let data_len = bit_vec.len();
        let half_field_size = self.field_size / 2;
        let tick_count = self.get_tick_count(bit_vec);

        for i in 0..tick_count {
            let start = i * half_field_size;
            let end = std::cmp::min(start + half_field_size, data_len);

            if let Some(prediction) = &mut self.prediction {
                prediction.add_tick(i);
            }

            self.push_to_buffer(bit_vec[start..end].to_vec());
        }

        for _ in 0..prediction_depth {
            self.push_to_buffer(vec![]);
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

    pub fn predict(&mut self, bit_vec: &[bool], prediction_depth: usize) -> Vec<Vec<bool>> {
        println!("===== PREDICTION =====");
        self.clean_neurons();

        let tick_count = self.get_tick_count(bit_vec);

        self.prediction = Some(PredictionProcessing::new(
            tick_count,
            self.computed_params.field_size,
            self.output_field_index,
            prediction_depth,
        ));

        self.push_data_and_apply(bit_vec, prediction_depth);

        let res = self.prediction.as_ref().unwrap().get_prediction();

        self.prediction = None;

        res
    }

    /**
     * Set all the values of neurons and refract intervals to 0
     */
    fn clean_neurons(&mut self) {
        let layer_size = self.layer_size;

        self.neurons_1 = vec![0u8; layer_size];
        self.neurons_2 = vec![0u8; layer_size];
        self.refract_intervals_1 = vec![0u8; layer_size];
        self.refract_intervals_2 = vec![0u8; layer_size];
    }

    pub fn get_output_field_state(&self) -> Vec<u8> {
        let mut res: Vec<u8> = vec![];

        for field_index in self.output_field_neuron_indexes.iter() {
            res.push(self.neurons_2[*field_index]);
        }

        res
    }

    pub fn print_states(&self) {
        println!("STATES:");
        println!();
        println!("LAYER 1:");
        self.print_state(&self.neurons_1, &self.refract_intervals_1);
        println!("LAYER 2:");
        self.print_state(&self.neurons_2, &self.refract_intervals_2);
        println!();
        println!();
    }

    fn print_state(&self, layer: &Vec<u8>, refract_intervals: &Vec<u8>) {
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

                        if layer[neuron_index] > 0 {
                            print!("+");
                        } else {
                            let refract_interval = refract_intervals[neuron_index];

                            if refract_interval > 0 {
                                print!("{}", refract_interval);
                            } else {
                                print!(".");
                            }
                        }
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

        refract_intervals[neuron_index]
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

        neurons[neuron_index]
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
        weights_layer: &Vec<f32>,
        neuron_x: usize,
        neuron_y: usize,
    ) -> Vec<f32> {
        let mut res = vec![0f32; self.computed_params.layer_size];

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

                        res[target_x + target_y * self.computed_params.row_width] = weights_layer
                            [target_neuron_index * self.computed_params.layer_size + neuron_index];
                    }
                }
            }
        }

        res
    }

    pub fn get_neuron_forward_synapses(
        &self,
        layer_index: u8,
        neuron_x: usize,
        neuron_y: usize,
    ) -> Vec<f32> {
        let weights_buffer = if layer_index == 1 {
            &self.buffer_forward_synapses_1_to_2
        } else {
            &self.buffer_forward_synapses_2_to_1
        };

        let layer_size = self.neurons_1.len();

        let mut weights_layer = vec![0.0_f32; layer_size * layer_size];

        // Читаем буфер обратно в data
        weights_buffer.read(&mut weights_layer).enq().unwrap();

        self.get_neuron_weights(&weights_layer, neuron_x, neuron_y)
    }

    pub fn get_neuron_distance_weights(
        &self,
        layer_index: u8,
        neuron_x: usize,
        neuron_y: usize,
    ) -> Vec<f32> {
        let weights_layer = if layer_index == 1 {
            &self.forward_distance_weights
        } else {
            &self.forward_distance_weights
        };

        self.get_neuron_weights(weights_layer, neuron_x, neuron_y)
    }

    pub fn get_forward_synapses_sum(&self, layer_index: u8) -> f32 {
        let weights_layer = if layer_index == 1 {
            &self.forward_synapses_1_to_2
        } else {
            &self.forward_synapses_2_to_1
        };

        weights_layer.into_iter().sum()
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

    fn apply_signal(&mut self, bit_vec: &[bool], counter: &u8) {
        let (apply_vec, rest) = self.split_signal(bit_vec);

        self.push_shift(apply_vec, true);

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
                let input_phase = self.input_phase;

                self.input_signal(bit_vec, input_phase);

                if *is_source {
                    if let Some(prediction) = &mut self.prediction {
                        prediction.add_tick_split(input_phase);
                    }
                }

                self.input_phase = match self.input_phase {
                    InputPhase::Even => InputPhase::Odd,
                    InputPhase::Odd => InputPhase::Even,
                };
            }
            Action::EmptyShift1to2 => {
                self.shift_1_to_2();

                let should_read = self
                    .prediction
                    .as_mut()
                    .map_or(false, |prediction| prediction.should_read());

                if should_read {
                    let last_field_state = self.get_output_field_state();

                    self.prediction.as_mut().unwrap().read(&last_field_state);
                }
            }
            Action::EmptyShift2to1 => {
                self.shift_2_to_1();

                if let Some(prediction) = &mut self.prediction {
                    if !prediction.is_finished() {
                        prediction.shift();
                    }
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

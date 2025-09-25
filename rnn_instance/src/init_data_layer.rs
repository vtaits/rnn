use std::{
    env,
    sync::{Arc, RwLock},
};

use data_streams::{train_network, ComplexStream, TrainingStream};
use rnn_core::{
    DataLayer, DataLayerParams, LayerParams, MultipleFileLogger, MultipleFileLoggerParams, Network,
    SynapseParams,
};
use timeline_helpers::{ComplexTimeline, ComplexTimelineValue, Timeline};

use crate::InitDataLayerParams;

pub fn init_data_layer(
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    timelines: Vec<Box<dyn Timeline>>,
    training_streams: Vec<Box<dyn TrainingStream>>,
    params: &InitDataLayerParams,
) -> (
    DataLayer<Vec<ComplexTimelineValue>>,
    Vec<(Vec<ComplexTimelineValue>, Vec<bool>)>,
) {
    let mut complex_stream = ComplexStream::new(training_streams);

    let complex_timeline = Arc::new(ComplexTimeline::new(timelines));

    let LayerParams {
        field_width,
        field_height,
        layer_height,
        layer_width,
        ..
    } = layer_params;

    let is_log_to_files = env::var("LOG_TO_FILES").map_or(false, |value| value == "1");

    let merged_synapse_params = if let Some(redefine_params) = &params.redefine_params {
        SynapseParams {
            alpha: redefine_params.alpha,
            gamma_inc: redefine_params.gamma_inc,
            gamma_dec: redefine_params.gamma_dec,
            g_dec: redefine_params.g_dec,
            g_inc: redefine_params.g_inc,
            g_0: redefine_params.g_0,
            min_g: synapse_params.min_g,
            max_g: synapse_params.max_g,
            h: redefine_params.h,
            threshold_train: redefine_params.threshold_train,
            threshold_predict_min: redefine_params.threshold_predict_min,
            threshold_predict_max: redefine_params.threshold_predict_max,
            refract_interval: synapse_params.refract_interval,
            signal_shift_interval: synapse_params.signal_shift_interval,
            signal_rest_shift_limit: synapse_params.signal_rest_shift_limit,
            signal_copy_shifts: synapse_params.signal_copy_shifts,
            excite_neuron_limit: synapse_params.excite_neuron_limit,
        }
    } else {
        synapse_params
    };

    let layer_params_with_partitions = LayerParams {
        field_width,
        field_height,
        layer_height,
        layer_width,
        partitions: Some(complex_timeline.get_partitions()),
    };

    let network = Network::new(
        layer_params_with_partitions,
        merged_synapse_params,
        if is_log_to_files {
            Some(Box::new(MultipleFileLogger::new(
                MultipleFileLoggerParams {
                    weights_diff_path_1: Some("diffs1.log"),
                    weights_diff_path_2: Some("diffs2.log"),
                    sum_path_1: Some("total1.log"),
                    sum_path_2: Some("total2.log"),
                    count_path: Some("count.log"),
                },
                field_width * field_height,
            )))
        } else {
            None
        },
    );

    let mut data_layer = DataLayer::new(
        DataLayerParams {
            data_to_binary: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move |data: Vec<ComplexTimelineValue>| complex_timeline.get_bits(&data))
            },
            binary_to_data: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move |binary| Ok(complex_timeline.reverse(&binary)))
            },
            get_target_mask: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move || complex_timeline.get_target_mask())
            },
            normalize_prediction: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move |data| complex_timeline.normalize_prediction(data))
            },
            regress: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(
                    move |original: &Vec<ComplexTimelineValue>,
                          computed: &Vec<ComplexTimelineValue>| {
                        complex_timeline.get_regress_difference(&original, &computed)
                    },
                )
            },
        },
        Arc::new(RwLock::new(network)),
    );

    let start_end_indexes = if let (Some(start), Some(end)) = (params.start_index, params.end_index)
    {
        Some((start, end))
    } else {
        None
    };

    let start_end_measure_indexes = if let (Some(start), Some(end)) =
        (params.start_measurement_index, params.end_measurement_index)
    {
        Some((start, end))
    } else {
        None
    };

    let measurement_data = train_network(
        &mut data_layer,
        &mut complex_stream,
        start_end_indexes,
        start_end_measure_indexes,
    );

    (data_layer, measurement_data)
}

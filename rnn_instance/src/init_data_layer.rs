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
) -> (DataLayer<Vec<ComplexTimelineValue>>, Vec<Vec<bool>>) {
    let mut complex_stream = ComplexStream::new(training_streams);

    let complex_timeline = Arc::new(ComplexTimeline::new(timelines));

    let LayerParams {
        field_width,
        field_height,
        ..
    } = layer_params;

    let is_log_to_files = env::var("LOG_TO_FILES").map_or(false, |value| value == "1");

    let network = Network::new(
        layer_params,
        synapse_params,
        if is_log_to_files {
            Some(Box::new(MultipleFileLogger::new(
                MultipleFileLoggerParams {
                    weights_diff_path_1: Some("diffs1.txt"),
                    weights_diff_path_2: Some("diffs2.txt"),
                    sum_path_1: Some("total1.txt"),
                    sum_path_2: Some("total2.txt"),
                    count_path: Some("count.txt"),
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
            get_target_mask: Box::new(move || complex_timeline.get_target_mask()),
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

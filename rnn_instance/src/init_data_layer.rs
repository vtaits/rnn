use std::{sync::Arc, sync::RwLock};

use data_streams::{train_network, ComplexStream, TrainingStream};
use rnn_core::{
    DataLayer, DataLayerParams, LayerParams, MultipleFileLogger, MultipleFileLoggerParams, Network,
    SynapseParams,
};
use timeline_helpers::{ComplexTimeline, ComplexTimelineValue, Timeline};

pub fn init_data_layer(
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    timelines: Vec<Box<dyn Timeline>>,
    training_streams: Vec<Box<dyn TrainingStream>>,
) -> DataLayer<Vec<ComplexTimelineValue>> {
    let mut complex_stream = ComplexStream::new(training_streams);

    let complex_timeline = Arc::new(ComplexTimeline::new(timelines));

    let LayerParams {
        field_width,
        field_height,
        ..
    } = layer_params;

    let network = Network::new(
        layer_params,
        synapse_params,
        Some(Box::new(MultipleFileLogger::new(
            MultipleFileLoggerParams {
                weights_diff_path_1: Some("diffs1.txt"),
                weights_diff_path_2: Some("diffs2.txt"),
                sum_path_1: Some("total1.txt"),
                sum_path_2: Some("total2.txt"),
                count_path: Some("count.txt"),
            },
            field_width * field_height,
        ))),
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
        },
        Arc::new(RwLock::new(network)),
    );

    train_network(&mut data_layer, &mut complex_stream);

    data_layer
}

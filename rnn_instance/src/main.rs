use console_ui::run_console_app;
use data_streams::{CsvStream, TrainingStream};
use rnn_core::{NetworkParams, SynapseParams};
use rnn_instance::init_network;
use timeline_helpers::{ComplexTimelineItem, FloatTimeline, FloatTimelineParams};

fn main() {
    let params = NetworkParams {
        field_width: 5,
        field_height: 4,
        layer_width: 5,
        layer_height: 5,
    };

    let synapse_params = SynapseParams {
        alpha: 20.0,
        gamma: 0.5,
        g_dec: 0.001,
        g_inc: 0.1,
        g_0: 1.0,
        initial_strong_g: 10.0,
        h: 2,
        refract_interval: 3,
        threshold: 0.9,
    };

    let timelines: Vec<ComplexTimelineItem> = vec![ComplexTimelineItem::Float(FloatTimeline::new(
        FloatTimelineParams {
            min_value: 0.0,
            max_value: 30.0,
            capacity: 16,
            get_multiplier: None,
            get_reverse_multiplier: None,
        },
    ))];

    let streams: Vec<Box<dyn TrainingStream>> = vec![Box::new(
        // CsvStream::new("../data_streams/src/training/csv_stream_test.csv").unwrap(),
        CsvStream::new("../media/csv/-2024-08-02.csv").unwrap(),
    )];

    let network = init_network(params, synapse_params, timelines, streams);

    let _ = run_console_app(Box::new(network));
}

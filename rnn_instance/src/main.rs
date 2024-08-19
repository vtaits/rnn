use console_ui::run_console_app;
use data_streams::{CsvStream, TrainingStream};
use rnn_core::{NetworkParams, SynapseParams};
use rnn_instance::init_network;
use timeline_helpers::{FloatTimeline, FloatTimelineParams, Timeline};

fn main() {
    let params = NetworkParams {
        field_width: 6,
        field_height: 6,
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

    let timelines: Vec<Box<dyn Timeline>> = vec![
        Box::new(FloatTimeline::new(FloatTimelineParams {
            min_value: 0.0,
            max_value: 30.0,
            capacity: 16,
            get_multiplier: None,
            get_reverse_multiplier: None,
        })),
        Box::new(FloatTimeline::new(FloatTimelineParams {
            min_value: 0.0,
            max_value: 30000.0,
            capacity: 16,
            get_multiplier: None,
            get_reverse_multiplier: None,
        })),
    ];

    let streams: Vec<Box<dyn TrainingStream>> = vec![
        Box::new(CsvStream::new("../media/csv/cpu_small.csv").unwrap()),
        Box::new(CsvStream::new("../media/csv/disk_busy_small.csv").unwrap()),
    ];
    let network = init_network(params, synapse_params, timelines, streams);

    let _ = run_console_app(Box::new(network));
}

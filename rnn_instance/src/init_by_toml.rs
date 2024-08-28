use std::{fs, path::Path};

use data_streams::{init_training_stream_by_config, TrainingStreamConfig};
use rnn_core::{DataLayer, LayerParams, SynapseParams};
use serde_derive::Deserialize;
use timeline_helpers::{init_timeline_by_config, ComplexTimelineValue, TimelineConfig};

use crate::init_data_layer;

#[derive(Deserialize)]
struct InitConfig {
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    training_streams: Vec<TrainingStreamConfig>,
    timelines: Vec<TimelineConfig>,
}

pub fn init_by_toml<P: AsRef<Path>>(file_path: P) -> DataLayer<Vec<ComplexTimelineValue>> {
    let toml_str = fs::read_to_string(file_path).expect("Failed to read TOML file");

    let config: InitConfig = toml::from_str(&toml_str).expect("Failed to parse TOML");

    let timelines = config
        .timelines
        .iter()
        .map(|timeline_config| init_timeline_by_config(timeline_config))
        .collect();

    let training_streams = config
        .training_streams
        .iter()
        .map(|training_stream_config| init_training_stream_by_config(training_stream_config))
        .collect();

    init_data_layer(
        config.layer_params,
        config.synapse_params,
        timelines,
        training_streams,
    )
}

use std::fs;

use data_streams::{init_training_stream_by_config, TrainingStreamConfig};
use rnn_core::{DataLayer, LayerParams, SynapseParams};
use serde_derive::Deserialize;
use timeline_helpers::{init_timeline_by_config, ComplexTimelineValue, TimelineConfig};

use crate::{get_file_path::get_file_path, init_data_layer, structs::InitDataLayerParams};

#[derive(Deserialize)]
struct InitConfig {
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    training_streams: Option<Vec<TrainingStreamConfig>>,
    timelines: Option<Vec<TimelineConfig>>,
}

pub fn init_by_toml(
    file_path: &str,
    config_dir: &Option<String>,
    params: &InitDataLayerParams,
) -> (DataLayer<Vec<ComplexTimelineValue>>, Vec<Vec<bool>>) {
    let train = params.train;

    let full_path = get_file_path(file_path, config_dir);

    let toml_str = fs::read_to_string(full_path).expect("Failed to read TOML file");

    let config: InitConfig = toml::from_str(&toml_str).expect("Failed to parse TOML");

    let timelines = config.timelines.map_or(vec![], |timelines| {
        timelines.iter().map(init_timeline_by_config).collect()
    });

    let training_streams = if train {
        config.training_streams.map_or(vec![], |training_streams| {
            training_streams
                .iter()
                .map(|config| init_training_stream_by_config(config, config_dir))
                .collect()
        })
    } else {
        vec![]
    };

    init_data_layer(
        config.layer_params,
        config.synapse_params,
        timelines,
        training_streams,
        params,
    )
}

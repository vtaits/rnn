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
    training_streams: Option<Vec<TrainingStreamConfig>>,
    timelines: Option<Vec<TimelineConfig>>,
}

pub fn init_by_toml(
    file_path: String,
    config_dir: &Option<String>,
) -> DataLayer<Vec<ComplexTimelineValue>> {
    let full_path = match config_dir {
        Some(config_dir) => {
            let path = Path::new(&config_dir).join(file_path);

            path.to_str().unwrap().to_string()
        }
        None => file_path,
    };

    let toml_str = fs::read_to_string(full_path).expect("Failed to read TOML file");

    let config: InitConfig = toml::from_str(&toml_str).expect("Failed to parse TOML");

    let timelines = config.timelines.map_or(vec![], |timelines| {
        timelines.iter().map(init_timeline_by_config).collect()
    });

    let training_streams = config.training_streams.map_or(vec![], |training_streams| {
        training_streams
            .iter()
            .map(|config| init_training_stream_by_config(config, config_dir))
            .collect()
    });

    init_data_layer(
        config.layer_params,
        config.synapse_params,
        timelines,
        training_streams,
    )
}

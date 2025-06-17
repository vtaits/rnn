use std::env;

use rnn_core::{DataLayer, Network};
use timeline_helpers::ComplexTimelineValue;

use crate::{get_file_path::get_file_path, init_by_toml, structs::InitDataLayerParams};

pub fn init_data_layer_by_env(
    params: &InitDataLayerParams,
) -> (DataLayer<Vec<ComplexTimelineValue>>, Vec<Vec<bool>>) {
    let InitDataLayerParams { train, end_measurement_index, start_measurement_index } = params;

    let config_dir: Option<String> = env::var("CONFIG_DIR").ok();
    let config_path = env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");

    let dump_gzip_path = env::var("DUMP_GZIP_PATH");
    let dump_path = env::var("DUMP_PATH");

    let (mut data_layer, measurement_data) = init_by_toml(
        &config_path,
        &config_dir,
        &InitDataLayerParams {
            train: *train && dump_gzip_path.is_err() && dump_path.is_err(),
            end_measurement_index: end_measurement_index.clone(),
            start_measurement_index: start_measurement_index.clone(),
        },
    );

    let gzip_dump_path = env::var("DUMP_GZIP_PATH").unwrap_or_default();
    let dump_path = env::var("DUMP_PATH").unwrap_or_default();

    if gzip_dump_path.len() > 0 {
        let full_gzip_dump_path = get_file_path(&gzip_dump_path, &config_dir);

        let dump = std::fs::read(full_gzip_dump_path).unwrap();

        let network = Network::from_gzip_dump_bytes(&dump).unwrap();

        data_layer.replace_network(network);
    } else if dump_path.len() > 0 {
        let full_dump_path = get_file_path(&dump_path, &config_dir);

        let dump = std::fs::read_to_string(full_dump_path).unwrap();

        let network = Network::from_json_dump(&dump).unwrap();

        data_layer.replace_network(network);
    }

    (data_layer, measurement_data)
}

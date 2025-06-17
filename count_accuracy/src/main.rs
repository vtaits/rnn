use std::env;

use tokio;

use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};

#[tokio::main]
async fn main() -> Result<(), ()> {
    let start_index = env::var("START_MEASURE_INDEX")
        .expect("START_MEASURE_INDEX should be setted")
        .parse::<usize>()
        .expect("START_MEASURE_INDEX should be a number");
    let end_index = env::var("END_MEASURE_INDEX")
        .expect("END_MEASURE_INDEX should be setted")
        .parse::<usize>()
        .expect("END_MEASURE_INDEX should be a number");

    let (mut data_layer, measurement_data) = init_data_layer_by_env(&InitDataLayerParams {
        train: true,
        start_index: None,
        end_index: None,
        end_measurement_index: Some(end_index),
        start_measurement_index: Some(start_index),
    });

    let (positive, negative) = data_layer.count_accuracy(measurement_data);

    println!(
        "positive: {}, negative: {}, total: {}",
        positive,
        negative,
        positive + negative
    );

    Ok(())
}

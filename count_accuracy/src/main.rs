use std::env;

use tokio;

use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};

#[tokio::main]
async fn main() -> Result<(), ()> {
    let measure_data_length = env::var("MEASURE_DATA_LENGTH")
        .expect("MEASURE_DATA_LENGTH should be setted")
        .parse::<usize>()
        .expect("MEASURE_DATA_LENGTH should be a number");
    let measures_count = env::var("MEASURES_COUNT")
        .expect("MEASURES_COUNT should be setted")
        .parse::<usize>()
        .expect("MEASURES_COUNT should be a number");

    let mut total_positive = 0;
    let mut total_negative = 0;

    for page in 0..measures_count {
        println!("Try #{}", page + 1);

        let start_index = page * measure_data_length;
        let end_index = start_index + measure_data_length;

        let (mut data_layer, measurement_data) = init_data_layer_by_env(&InitDataLayerParams {
            train: true,
            start_index: None,
            end_index: None,
            end_measurement_index: Some(end_index),
            start_measurement_index: Some(start_index),
        });

        let (positive, negative) = data_layer.count_accuracy(measurement_data);

        total_positive += positive;
        total_negative += negative;

        println!(
            "positive: {}, negative: {}, total: {}",
            positive,
            negative,
            positive + negative
        );
    }

    println!("Total");

    println!(
        "positive: {}, negative: {}, total: {}",
        total_positive,
        total_negative,
        total_positive + total_negative
    );

    Ok(())
}

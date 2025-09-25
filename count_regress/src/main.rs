use std::env;

use rnn_core::RegressResult;
use tokio;

use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};

fn mape(results: &[RegressResult]) -> f32 {
    let mut sum = 0.0;
    let mut count = 0;
    for r in results {
        if r.actual != 0.0 {
            sum += ((r.actual - r.received).abs() / r.actual.abs());
            count += 1;
        }
    }

    if count == 0 {
        panic!("There are no results")
    }

    sum / count as f32 * 100.0
}

fn wape(results: &[RegressResult]) -> f32 {
    let abs_error_sum: f32 = results.iter().map(|r| (r.actual - r.received).abs()).sum();
    let actual_sum: f32 = results.iter().map(|r| r.actual.abs()).sum();

    if actual_sum == 0.0 {
        panic!("There are no results");
    }

    abs_error_sum / actual_sum * 100.0
}

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

    let mut total_res = vec![];

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
            redefine_params: None,
        });

        let res = data_layer.regress(measurement_data);

        println!("MAPE = {}, WAPE = {}", mape(&res), wape(&res));

        for regress_result in res {
            println!(
                "actual: {} ; received: {} ; diff: {}",
                regress_result.actual, regress_result.received, regress_result.diff
            );

            total_res.push(regress_result);
        }
    }

    println!(
        "TOTAL MAPE = {}, TOTAL WAPE = {}",
        mape(&total_res),
        wape(&total_res)
    );

    Ok(())
}

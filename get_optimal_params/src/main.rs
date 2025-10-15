use std::env;

use rnn_core::CountAccuracyResult;
use tokio;

use itertools::Itertools;
use rnn_instance::{init_data_layer_by_env, InitDataLayerParams, RedefineParams};

fn frange(start: f32, end: f32, step: f32) -> Vec<f32> {
    let mut v = Vec::new();
    let mut val = start;
    while val <= end {
        v.push(val);
        val += step;
    }
    v
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

    let ranges = vec![
        // alpha
        frange(1.95, 2.051, 0.025),
        // frange(1.935, 1.945, 0.001),
        // vec![1.94],
        // gamma_dec
        // frange(0.3, 0.7, 0.1),
        vec![0.6],
        // gamma_inc
        // frange(0.3, 0.7, 0.1),
        vec![0.4],
        // g_dec
        // frange(1.0, 3.01, 1.0),
        vec![1.0],
        // g_inc
        // frange(1.0, 3.01, 1.0),
        vec![1.0],
        // g_0
        // frange(0.0, 3.01, 1.0),
        vec![1.0],
        // h = 1.4
        frange(0.9, 1.11, 0.05),
        // frange(1.015, 1.025, 0.001),
        // vec![1.0],
        // refract_interval = 2
        // frange(1.0, 3.01, 1.0),
        // threshold_predict_min
        // frange(0.67, 0.731, 0.01),
        vec![0.7],
        // threshold_predict_max
        // frange(0.8, 0.881, 0.01),
        vec![0.85],
        // signal_shift_interval = 1
        // frange(1.0, 3.01, 1.0),
    ];

    let iterators = ranges.iter().map(|v| v.iter().cloned()).collect::<Vec<_>>();

    for combo in iterators.into_iter().multi_cartesian_product() {
        let alpha = combo[0];
        let gamma_dec = combo[1];
        let gamma_inc = combo[2];
        let g_dec = combo[3];
        let g_inc = combo[4];
        let g_0 = combo[5];
        let h = combo[6];
        let threshold_predict_min = combo[7];
        let threshold_predict_max = combo[8];

        if threshold_predict_max > threshold_predict_min {
            let redefine_params = RedefineParams {
                alpha,
                gamma_dec,
                gamma_inc,
                g_dec,
                g_inc,
                g_0,
                h,
                threshold_predict_min,
                threshold_predict_max,
            };

            let mut total_positive = 0;
            let mut total_negative = 0;

            let mut true_positive_neurons = 0usize;
            let mut true_negative_neurons = 0usize;
            let mut false_positive_neurons = 0usize;
            let mut false_negative_neurons = 0usize;

            for page in 0..measures_count {
                let redefine_params = redefine_params.clone();

                let start_index = page * measure_data_length;
                let end_index = start_index + measure_data_length;

                let (mut data_layer, measurement_data) =
                    init_data_layer_by_env(&InitDataLayerParams {
                        train: true,
                        start_index: None,
                        end_index: None,
                        end_measurement_index: Some(end_index),
                        start_measurement_index: Some(start_index),
                        redefine_params: Some(redefine_params),
                    });

                let CountAccuracyResult {
                    positive,
                    negative,
                    true_positive,
                    true_negative,
                    false_positive,
                    false_negative,
                } = data_layer.count_accuracy(measurement_data);

                total_positive += positive;
                total_negative += negative;
                true_positive_neurons += true_positive;
                true_negative_neurons += true_negative;
                false_positive_neurons += false_positive;
                false_negative_neurons += false_negative;
            }

            if total_positive > 50 {
                println!("{:?}", redefine_params);

                println!(
                    "positive: {}, negative: {}, total: {}, true positive: {}, true negative: {}, false positive: {}, false negative: {}",
                    total_positive,
                    total_negative,
                    total_positive + total_negative,
                    true_positive_neurons,
                    true_negative_neurons,
                    false_positive_neurons,
                    false_negative_neurons,
                );

                println!();
                println!("=======================");
                println!();
            }
        }
    }

    Ok(())
}

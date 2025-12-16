use std::sync::RwLock;
use std::time::Instant;
use std::{env, sync::Arc};

use rnn_core::{DataLayer, DataLayerParams, LayerParams, Network, SynapseParams};
use timeline_helpers::{
    ComplexTimeline, ComplexTimelineValue, FloatTimeline, FloatTimelineParams, Timeline,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), ()> {
    let streams_length = env::var("STREAMS_LENGTH")
        .expect("STREAMS_LENGTH should be setted")
        .parse::<usize>()
        .expect("STREAMS_LENGTH should be a number");
    let stream_size = env::var("STREAM_SIZE")
        .expect("STREAM_SIZE should be setted")
        .parse::<u8>()
        .expect("STREAM_SIZE should be a number");
    let field_count = env::var("FIELD_COUNT")
        .expect("FIELD_COUNT should be setted")
        .parse::<usize>()
        .expect("FIELD_COUNT should be a number");
    let train_data_length = env::var("TRAIN_DATA_LENGTH")
        .expect("TRAIN_DATA_LENGTH should be setted")
        .parse::<usize>()
        .expect("TRAIN_DATA_LENGTH should be a number");
    let measure_data_length = env::var("MEASURE_DATA_LENGTH")
        .expect("MEASURE_DATA_LENGTH should be setted")
        .parse::<usize>()
        .expect("MEASURE_DATA_LENGTH should be a number");
    let measures_count = env::var("MEASURES_COUNT")
        .expect("MEASURES_COUNT should be setted")
        .parse::<usize>()
        .expect("MEASURES_COUNT should be a number");

    let mut toal_duration_mills = 0;

    let total_size = streams_length * (stream_size as usize) * field_count * 2;

    for page in 0..measures_count {
        println!("Try #{}", page + 1);

        let params = LayerParams {
            field_width: stream_size as usize * 2,
            field_height: streams_length,
            layer_width: field_count,
            layer_height: 1,
            partitions: None,
        };

        let synapse_params = SynapseParams {
            alpha: 2.0,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 0.05,
            g_inc: 0.1,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 0.5,
            refract_interval: 0,
            threshold_predict_max: 0.9,
            threshold_predict_min: 0.9,
            signal_shift_interval: 0,
            signal_rest_shift_limit: Some(0),
            excite_neuron_limit: 0.8,
        };

        let start_init = Instant::now();

        println!("Initialization {}", total_size);

        let network = Network::new(params, synapse_params, None);

        let end_init = start_init.elapsed();

        println!("Network is ready");
        println!("Initialization time (ms): {}", end_init.as_millis());

        let mut timelines: Vec<Box<dyn Timeline>> = vec![];

        for _ in 0..streams_length {
            timelines.push(Box::new(FloatTimeline::new(FloatTimelineParams {
                is_single_bit: true,
                min_value: 0.0,
                max_value: 1.0,
                capacity: stream_size,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: true,
            })));
        }

        let complex_timeline = Arc::new(ComplexTimeline::new(timelines));

        let mut data_layer = DataLayer::new(
            DataLayerParams {
                data_to_binary: {
                    let complex_timeline = Arc::clone(&complex_timeline);

                    Box::new(move |data: Vec<ComplexTimelineValue>| {
                        complex_timeline.get_bits(&data)
                    })
                },
                binary_to_data: {
                    let complex_timeline = Arc::clone(&complex_timeline);

                    Box::new(move |binary| Ok(complex_timeline.reverse(&binary)))
                },
                get_target_mask: {
                    let complex_timeline = Arc::clone(&complex_timeline);

                    Box::new(move || complex_timeline.get_target_mask())
                },
                normalize_prediction: {
                    let complex_timeline = Arc::clone(&complex_timeline);

                    Box::new(move |data| complex_timeline.normalize_prediction(data))
                },
                regress: {
                    let complex_timeline = Arc::clone(&complex_timeline);

                    Box::new(
                        move |original: &Vec<ComplexTimelineValue>,
                              computed: &Vec<ComplexTimelineValue>| {
                            complex_timeline.get_regress_difference(&original, &computed)
                        },
                    )
                },
            },
            Arc::new(RwLock::new(network)),
        );

        let mut measurement_data = vec![];

        for _ in 0..=measure_data_length {
            let mut values = vec![];

            for _ in 0..streams_length {
                values.push(ComplexTimelineValue::Float(0.5));
            }

            measurement_data.push((values, vec![]));
        }

        let start_training = Instant::now();

        for _ in 0..train_data_length {
            data_layer.push_data_and_apply(vec![], 0);
        }

        let duration_training = start_training.elapsed().as_millis();

        let start_prediction = Instant::now();

        let _ = data_layer.regress_deep(measurement_data);

        let duration_prediction = start_prediction.elapsed().as_millis();

        let duration_mills = duration_training + duration_prediction;
        toal_duration_mills += duration_mills;

        println!(
            "Training time (ms): {} , Prediction time (ms): {} , Total time (ms): {}",
            duration_training, duration_prediction, duration_mills
        );
    }

    println!(
        "AVERAGE EXECUTION TIME = {}",
        toal_duration_mills / measures_count as u128,
    );

    Ok(())
}

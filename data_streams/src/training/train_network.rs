use rnn_core::DataLayer;
use timeline_helpers::ComplexTimelineValue;

use super::ComplexStream;

pub fn train_network(
    data_layer: &mut DataLayer<Vec<ComplexTimelineValue>>,
    complex_stream: &mut ComplexStream,
    start_end_indexes: Option<(usize, usize)>,
    start_end_measure_indexes: Option<(usize, usize)>,
) -> Vec<(Vec<ComplexTimelineValue>, Vec<bool>)> {
    let mut index: usize = 0;

    let mut measurement_data = vec![];

    while !complex_stream.is_finish() {
        let data = complex_stream.get_value();

        let is_handle = if let Some((start_index, end_index)) = start_end_indexes {
            start_index <= index && index < end_index
        } else {
            true
        };

        if is_handle {
            if let Some((start_measure_index, end_measure_index)) = start_end_measure_indexes {
                if start_measure_index <= index && index < end_measure_index {
                    if let Ok(measurement_vec) = data_layer.process_for_measure(data.clone()) {
                        measurement_data.push((data, measurement_vec));
                    }
                } else {
                    data_layer.push_data_and_apply(data, 0);
                }
            } else {
                data_layer.push_data_and_apply(data, 0);
            }
        }

        complex_stream.step();

        index += 1;
    }

    measurement_data
}

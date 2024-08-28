use rnn_core::DataLayer;
use timeline_helpers::ComplexTimelineValue;

use super::ComplexStream;

pub fn train_network(
    data_layer: &mut DataLayer<Vec<ComplexTimelineValue>>,
    complex_stream: &mut ComplexStream,
) {
    while !complex_stream.is_finish() {
        let data = complex_stream.get_value();

        data_layer.push_data(data);

        complex_stream.step();
    }
}
